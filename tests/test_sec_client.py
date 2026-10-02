import json
import time
from pathlib import Path

import httpx
import pytest

from app.clients import sec
from app.config import settings

SEC_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "company_tickers_exchange.json"
SUBMISSIONS_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "submissions.json"


def test_get_company_tickers_parses_and_saves_raw_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        calls.append((url, kwargs))
        return httpx.Response(
            200, content=SEC_FIXTURE_PATH.read_bytes(), request=httpx.Request("GET", url)
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))

    companies = sec.get_company_tickers()

    # The right URL was requested, with the User-Agent from settings
    url, kwargs = calls[0]
    assert url == "https://www.sec.gov/files/company_tickers_exchange.json"
    assert kwargs["headers"] == {"User-Agent": settings.SEC_USER_AGENT}

    # Every row has the four fields and a 10-character zero-padded CIK
    assert len(companies) == 8
    for company in companies:
        assert set(company) == {"cik", "ticker", "name", "exchange"}
        assert len(company["cik"]) == 10
    apple = next(company for company in companies if company["ticker"] == "AAPL")
    assert apple == {
        "cik": "0000320193",
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "exchange": "Nasdaq",
    }
    # A null exchange in the file stays None
    johnson = next(company for company in companies if company["ticker"] == "JNJ")
    assert johnson["exchange"] is None

    # The raw response was saved before parsing, byte for byte
    raw_file = tmp_path / "sec" / "company_tickers_exchange.json"
    assert raw_file.read_bytes() == SEC_FIXTURE_PATH.read_bytes()


def test_get_company_tickers_raises_on_http_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(403, content=b"Forbidden", request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))

    with pytest.raises(httpx.HTTPStatusError):
        sec.get_company_tickers()

    # Nothing is saved when the download failed
    assert not (tmp_path / "sec").exists()


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    # A fake clock: sleeping moves the clock forward instead of waiting. Returns the list of
    # sleeps, so tests can check how long the client waited
    clock = [1000.0]
    slept: list[float] = []

    def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(time, "sleep", fake_sleep)
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sec, "last_request_time", 0.0)
    return slept


def mock_httpx_get(
    monkeypatch: pytest.MonkeyPatch, answers: list[httpx.Response | Exception]
) -> list[tuple[str, dict]]:
    # Each call to httpx.get returns (or raises) the next answer. Returns the recorded calls
    calls: list[tuple[str, dict]] = []

    def fake_get(url: str, **kwargs: object) -> httpx.Response:
        calls.append((url, kwargs))
        answer = answers[len(calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(httpx, "get", fake_get)
    return calls


def make_response(status_code: int, content: bytes = b"{}") -> httpx.Response:
    return httpx.Response(
        status_code, content=content, request=httpx.Request("GET", "https://example.test")
    )


def test_sec_get_sends_the_user_agent(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    calls = mock_httpx_get(monkeypatch, [make_response(200)])

    sec.sec_get("https://example.test/file", timeout=12.0)

    url, kwargs = calls[0]
    assert url == "https://example.test/file"
    assert kwargs["headers"] == {"User-Agent": settings.SEC_USER_AGENT}
    assert kwargs["timeout"] == 12.0


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_sec_get_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], status_code: int
) -> None:
    calls = mock_httpx_get(monkeypatch, [make_response(status_code), make_response(200, b"ok")])

    response = sec.sec_get("https://example.test/file", timeout=30.0)

    assert response.content == b"ok"
    assert len(calls) == 2
    assert sleeps == [1]


@pytest.mark.parametrize("error", [httpx.ReadTimeout("slow"), httpx.ConnectError("refused")])
def test_sec_get_retries_timeouts_and_connection_errors(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], error: Exception
) -> None:
    calls = mock_httpx_get(monkeypatch, [error, make_response(200)])

    sec.sec_get("https://example.test/file", timeout=30.0)

    assert len(calls) == 2
    assert sleeps == [1]


def test_sec_get_gives_up_after_the_last_retry(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    calls = mock_httpx_get(monkeypatch, [make_response(503)] * 4)

    with pytest.raises(httpx.HTTPStatusError):
        sec.sec_get("https://example.test/file", timeout=30.0)

    # The first try plus 3 retries, with waits of 1, 2 and 4 seconds
    assert len(calls) == 4
    assert sleeps == [1, 2, 4]


def test_sec_get_raises_the_network_error_after_the_last_retry(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    calls = mock_httpx_get(monkeypatch, [httpx.ConnectError("refused")] * 4)

    with pytest.raises(httpx.ConnectError):
        sec.sec_get("https://example.test/file", timeout=30.0)

    assert len(calls) == 4
    assert sleeps == [1, 2, 4]


def test_sec_get_does_not_retry_a_404(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    calls = mock_httpx_get(monkeypatch, [make_response(404)])

    with pytest.raises(httpx.HTTPStatusError):
        sec.sec_get("https://example.test/file", timeout=30.0)

    assert len(calls) == 1
    assert sleeps == []


def test_sec_get_throttles_back_to_back_calls(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    mock_httpx_get(monkeypatch, [make_response(200), make_response(200)])

    sec.sec_get("https://example.test/one", timeout=30.0)
    # The first call needs no wait. The second one starts 0 seconds after it, so it waits 0.2
    assert sleeps == []
    sec.sec_get("https://example.test/two", timeout=30.0)
    assert sleeps == [0.2]


def test_get_submissions_uses_the_padded_cik_and_saves_the_raw_file(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], tmp_path: Path
) -> None:
    calls = mock_httpx_get(monkeypatch, [make_response(200, SUBMISSIONS_FIXTURE_PATH.read_bytes())])
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))

    submissions = sec.get_submissions("0000320193")

    assert calls[0][0] == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert submissions == json.loads(SUBMISSIONS_FIXTURE_PATH.read_text())
    raw_file = tmp_path / "sec" / "submissions" / "CIK0000320193.json"
    assert raw_file.read_bytes() == SUBMISSIONS_FIXTURE_PATH.read_bytes()


def test_get_submissions_can_fetch_an_older_page(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], tmp_path: Path
) -> None:
    calls = mock_httpx_get(monkeypatch, [make_response(200, b'{"form": []}')])
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))

    page = sec.get_submissions("0000019617", page_name="CIK0000019617-submissions-001.json")

    assert calls[0][0] == "https://data.sec.gov/submissions/CIK0000019617-submissions-001.json"
    assert page == {"form": []}
    assert (tmp_path / "sec" / "submissions" / "CIK0000019617-submissions-001.json").exists()


def test_download_filing_document_builds_the_url_and_saves_the_file(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], tmp_path: Path
) -> None:
    calls = mock_httpx_get(monkeypatch, [make_response(200, b"<html>10-K</html>")])
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))

    relative_path = sec.download_filing_document(
        "0000320193", "0000320193-25-000079", "aapl-20250927.htm"
    )

    # The URL has the CIK without leading zeros and the accession number without dashes
    url, kwargs = calls[0]
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
    )
    assert kwargs["timeout"] == 60.0
    # The file is saved under the padded CIK and the dashed accession number
    assert relative_path == "sec/filings/0000320193/0000320193-25-000079/aapl-20250927.htm"
    assert (tmp_path / relative_path).read_bytes() == b"<html>10-K</html>"


def test_get_company_facts_uses_the_padded_cik_and_saves_the_raw_file(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], tmp_path: Path
) -> None:
    body = b'{"cik": 320193, "entityName": "Apple Inc.", "facts": {"us-gaap": {}}}'
    calls = mock_httpx_get(monkeypatch, [make_response(200, body)])
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(tmp_path))

    facts = sec.get_company_facts("0000320193")

    url, kwargs = calls[0]
    assert url == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    assert kwargs["timeout"] == 60.0
    assert kwargs["headers"] == {"User-Agent": settings.SEC_USER_AGENT}
    assert facts == json.loads(body)
    # The raw file is saved before parsing, byte for byte
    raw_file = tmp_path / "sec" / "companyfacts" / "CIK0000320193.json"
    assert raw_file.read_bytes() == body
