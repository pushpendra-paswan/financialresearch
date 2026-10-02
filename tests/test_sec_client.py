from pathlib import Path

import httpx
import pytest

from app.clients import sec
from app.config import settings

SEC_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "company_tickers_exchange.json"


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
