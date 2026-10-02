# Run with: docker compose exec api python -m scripts.run_financial_ingestion
import logging
import sys

from app.database import SessionLocal
from app.exceptions import ConflictError
from app.services import financials

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

db = SessionLocal()
try:
    run = financials.ingest_financial_facts(db)
    print(f"Status: {run.status}")
    print(f"Message: {run.message}")
except ConflictError as error:
    print(error.message)
    sys.exit(1)
finally:
    db.close()
