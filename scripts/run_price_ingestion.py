# Run with: docker compose exec api python -m scripts.run_price_ingestion
# The first run on an empty price_bars table is the backfill
import logging
import sys

from app.database import SessionLocal
from app.exceptions import ConflictError
from app.services import prices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

db = SessionLocal()
try:
    run = prices.ingest_prices(db)
    print(f"Status: {run.status}")
    print(f"Message: {run.message}")
except ConflictError as error:
    print(error.message)
    sys.exit(1)
finally:
    db.close()
