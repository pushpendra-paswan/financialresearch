import logging
import time
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{file_name}"
FILING_DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# The SEC allows at most 10 requests per second; we stay at 5
MIN_SECONDS_BETWEEN_REQUESTS = 0.2
# Waits (in seconds) before retry 1, 2 and 3. After the third retry the error is raised
RETRY_WAITS = [1, 2, 4]

# Time of the last request, kept per process. The worker runs with --concurrency=1 so this
# throttle is effectively global
last_request_time = 0.0


def sec_get(url: str, timeout: float) -> httpx.Response:
    # The only function that calls httpx. Every SEC request goes through it, so the User-Agent,
    # the throttle and the retries are written once.
    global last_request_time

    attempt = 0
    while True:
        # 1. Throttle: wait until at least 0.2 seconds have passed since the last request
        seconds_to_wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - last_request_time)
        if seconds_to_wait > 0:
            time.sleep(seconds_to_wait)
        last_request_time = time.monotonic()

        # 2. Send the request. The SEC requires a User-Agent that identifies us
        try:
            response = httpx.get(
                url, headers={"User-Agent": settings.SEC_USER_AGENT}, timeout=timeout
            )
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            # Transient network problem: retry, or give up after the last retry
            if attempt == len(RETRY_WAITS):
                raise
            logger.warning("SEC request failed (%s), retry %d: %s", error, attempt + 1, url)
            time.sleep(RETRY_WAITS[attempt])
            attempt += 1
            continue

        # 3. HTTP 429 and 5xx are transient: retry. Other 4xx (a 404, for example) fail at once
        if response.status_code == 429 or response.status_code >= 500:
            if attempt == len(RETRY_WAITS):
                response.raise_for_status()
            logger.warning("SEC returned %d, retry %d: %s", response.status_code, attempt + 1, url)
            time.sleep(RETRY_WAITS[attempt])
            attempt += 1
            continue

        response.raise_for_status()
        return response


def get_company_tickers() -> list[dict]:
    response = sec_get(COMPANY_TICKERS_URL, timeout=30.0)

    # Save the raw file before parsing, so it can be re-processed without downloading again
    raw_path = Path(settings.RAW_DATA_DIR) / "sec" / "company_tickers_exchange.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(response.content)
    logger.info("Saved SEC company tickers to %s", raw_path)

    # The file is columnar: {"fields": ["cik", "name", "ticker", "exchange"], "data": [[...], ...]}
    # and the cik is a plain integer. Rows are read by field name, not by position.
    payload = response.json()
    fields = payload["fields"]
    companies = []
    for row in payload["data"]:
        values = dict(zip(fields, row, strict=True))
        companies.append(
            {
                # data.sec.gov endpoints need the CIK as 10 characters, padded with zeros
                "cik": str(values["cik"]).zfill(10),
                "ticker": values["ticker"],
                "name": values["name"],
                "exchange": values["exchange"],
            }
        )
    return companies


def get_submissions(cik: str, page_name: str | None = None) -> dict:
    # Without page_name: the company's main submissions file (company info plus the "recent"
    # filings block). With page_name (a name from filings.files in the main file): one older page
    # of filings. Both are saved under the same folder, so they are named differently.
    file_name = page_name or f"CIK{cik}.json"
    response = sec_get(SUBMISSIONS_URL.format(file_name=file_name), timeout=30.0)

    # Save the raw file before parsing (overwritten on every run)
    raw_path = Path(settings.RAW_DATA_DIR) / "sec" / "submissions" / file_name
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(response.content)
    logger.info("Saved SEC submissions to %s", raw_path)

    return response.json()


def download_filing_document(cik: str, accession_number: str, primary_document: str) -> str:
    # The URL needs the CIK WITHOUT leading zeros and the accession number WITHOUT dashes
    url = FILING_DOCUMENT_URL.format(
        cik=cik.lstrip("0"),
        accession=accession_number.replace("-", ""),
        document=primary_document,
    )
    response = sec_get(url, timeout=60.0)

    # Save the raw document before anything parses it. The folder uses the padded CIK and the
    # dashed accession number
    raw_dir = Path(settings.RAW_DATA_DIR)
    file_path = raw_dir / "sec" / "filings" / cik / accession_number / primary_document
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(response.content)
    logger.info("Saved filing document to %s", file_path)

    # The database stores the path relative to RAW_DATA_DIR, so moving the folder breaks nothing
    return str(file_path.relative_to(raw_dir))
