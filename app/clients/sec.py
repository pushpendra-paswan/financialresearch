import logging
from pathlib import Path

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


def get_company_tickers() -> list[dict]:
    # The SEC requires a User-Agent that identifies us on every request
    response = httpx.get(
        COMPANY_TICKERS_URL,
        headers={"User-Agent": settings.SEC_USER_AGENT},
        timeout=30.0,
    )
    response.raise_for_status()

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
