# Run with: docker compose exec api python -m scripts.seed_companies
import logging
import sys

from app.database import SessionLocal
from app.services import companies

# Only one ticker per company: GOOGL and GOOG are the same company (same CIK), so only GOOGL
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "ORCL", "CRM",
    "ADBE", "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "JNJ", "PFE", "UNH", "LLY", "MRK",
    "XOM", "CVX", "WMT", "KO", "PEP", "MCD", "NKE", "DIS", "BA", "CAT",
]  # fmt: skip

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

db = SessionLocal()
try:
    companies.seed_companies(db, TICKERS)
except ValueError as error:
    logger.error("Seed failed, nothing was written: %s", error)
    sys.exit(1)
finally:
    db.close()
