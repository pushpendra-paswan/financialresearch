import logging
from datetime import UTC, date, datetime, timedelta

import httpx
from sqlalchemy.orm import Session

from app.clients import sec
from app.config import settings
from app.exceptions import ConflictError
from app.models.ingestion import IngestionRun, IngestionStatus
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository
from app.repositories import ingestion as ingestion_repository

logger = logging.getLogger(__name__)

JOB_TYPE = "ingest_filings"
# Amendments (10-K/A, 10-Q/A), 8-Ks and everything else are ignored for now
STORED_FORM_TYPES = {"10-K", "10-Q"}
# A "running" row older than this is treated as left behind by a crashed run
STALE_RUN_AFTER = timedelta(hours=2)


# The three helpers below are shared by every ingestion job (filings, financial facts, prices).
# They only do the run bookkeeping; the jobs themselves stay long and sequential.
def start_run(db: Session, job_type: str, label: str) -> IngestionRun:
    # Do not start while another run of the same job is in progress
    running_since = datetime.now(UTC) - STALE_RUN_AFTER
    if ingestion_repository.get_recent_running_run(db, job_type, running_since):
        raise ConflictError(f"A {label} ingestion run is already in progress")

    # Record the run and commit at once, so it is visible while the job is still working
    run = ingestion_repository.create_run(db, job_type)
    db.commit()
    return run


def finish_run(db: Session, run: IngestionRun, failed_count: int, message: str) -> IngestionRun:
    # "partial" means at least one company or document failed
    if failed_count:
        run.status = IngestionStatus.partial
    else:
        run.status = IngestionStatus.success
    run.finished_at = datetime.now(UTC)
    run.message = message
    db.commit()
    return run


def fail_run(db: Session, run: IngestionRun, error: Exception) -> None:
    # Whatever went wrong, the run row must not stay "running". The caller re-raises the error
    db.rollback()
    run.status = IngestionStatus.failed
    run.finished_at = datetime.now(UTC)
    run.error = f"{type(error).__name__}: {error}"
    db.commit()


def ingest_filings(db: Session) -> IngestionRun:
    # 1-2. Guard against an overlapping run and record this one
    run = start_run(db, JOB_TYPE, "filing")

    # The one broad except in the project: whatever goes wrong, the run row must not stay
    # "running" forever. It records the failure and re-raises.
    try:
        # 3. Only filings filed on or after this date are stored. timedelta (not date.replace)
        # avoids the error for a leap day
        cutoff = date.today() - timedelta(days=365 * settings.FILINGS_LOOKBACK_YEARS)

        companies_processed = 0
        failed_tickers: list[str] = []
        new_filings = 0
        documents_downloaded = 0
        documents_failed = 0

        for company in company_repository.list_all(db):
            # Read these now: a rollback below expires the company object
            company_id = company.id
            ticker = company.ticker
            cik = company.cik

            # 4. Fetch the filing list and store the new 10-K and 10-Q filings. A failure here
            # skips only this company. KeyError and ValueError (bad JSON or dates) are caught
            # as well, so one malformed answer from the SEC cannot stop the other companies.
            try:
                submissions = sec.get_submissions(cik)
                company.industry = submissions.get("sicDescription") or None

                # The filings are in parallel lists (one list per field), in blocks. The
                # "recent" block holds at least the last year of filings. For companies that file
                # very often (banks) that is less than the lookback window, so in that case the
                # older pages that overlap the window are fetched as well
                recent = submissions["filings"]["recent"]
                blocks = [recent]
                oldest_recent = min(recent["filingDate"], default="")
                if oldest_recent > cutoff.isoformat():
                    for older_page in submissions["filings"]["files"]:
                        if date.fromisoformat(older_page["filingTo"]) >= cutoff:
                            blocks.append(sec.get_submissions(cik, page_name=older_page["name"]))

                stored_accession_numbers = filing_repository.get_accession_numbers(db, company_id)
                company_new_filings = 0
                for block in blocks:
                    rows = zip(
                        block["accessionNumber"],
                        block["filingDate"],
                        block["reportDate"],
                        block["form"],
                        block["primaryDocument"],
                        strict=True,
                    )
                    for accession_number, filing_date, report_date, form, primary_document in rows:
                        if form not in STORED_FORM_TYPES:
                            continue
                        filed_on = date.fromisoformat(filing_date)
                        if filed_on < cutoff:
                            continue
                        if accession_number in stored_accession_numbers:
                            continue
                        if not primary_document:
                            logger.warning(
                                "%s %s has no primary document, skipped", ticker, accession_number
                            )
                            continue

                        # The SEC sends an empty string when there is no report date
                        report_on = date.fromisoformat(report_date) if report_date else None
                        filing_repository.create(
                            db,
                            company_id,
                            accession_number,
                            form,
                            filed_on,
                            report_on,
                            report_on.year if report_on else None,
                            primary_document,
                        )
                        stored_accession_numbers.add(accession_number)
                        company_new_filings += 1

                # One commit per company, so a later failure never loses finished companies
                db.commit()
                companies_processed += 1
                new_filings += company_new_filings
                logger.info("%s: %d new filings", ticker, company_new_filings)
            except (httpx.HTTPError, OSError, KeyError, ValueError) as error:
                db.rollback()
                logger.error("Filing ingestion failed for %s: %s", ticker, error)
                failed_tickers.append(ticker)
                continue

            # 5. Download the documents of this company that are not on disk yet. This includes
            # documents that failed in an earlier run. One failure skips only that document
            for filing in filing_repository.list_without_document(db, company_id):
                try:
                    filing.raw_path = sec.download_filing_document(
                        cik, filing.accession_number, filing.primary_document
                    )
                    # Commit after each document, so a crash never loses finished downloads
                    db.commit()
                    documents_downloaded += 1
                except (httpx.HTTPError, OSError) as error:
                    logger.error(
                        "Document download failed for %s %s: %s",
                        ticker,
                        filing.accession_number,
                        error,
                    )
                    documents_failed += 1

        # 6. Finish the run. "partial" means at least one company or document failed
        message = (
            f"Companies processed: {companies_processed}, companies failed: {len(failed_tickers)}, "
            f"new filings: {new_filings}, documents downloaded: {documents_downloaded}, "
            f"documents failed: {documents_failed}"
        )
        if failed_tickers:
            message += f". Failed companies: {', '.join(failed_tickers)}"
        finish_run(db, run, len(failed_tickers) + documents_failed, message)
        logger.info("Filing ingestion finished (%s): %s", run.status, run.message)
        return run
    except Exception as error:
        # 7. Record the failure and re-raise it
        fail_run(db, run, error)
        logger.exception("Filing ingestion failed")
        raise
