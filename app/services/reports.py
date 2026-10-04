import re

from sqlalchemy.orm import Session

from app.config import settings
from app.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.models.chunks import DocumentChunk
from app.models.companies import Company
from app.models.reports import Report, ReportCitation
from app.models.users import UserRole
from app.repositories import agent as agent_repository
from app.repositories import audit as audit_repository
from app.repositories import chunks as chunk_repository
from app.repositories import companies as company_repository
from app.repositories import reports as report_repository
from app.repositories import users as user_repository
from app.schemas.reports import (
    ReportCitationResponse,
    ReportCompanyResponse,
    ReportDetail,
    ReportListResponse,
    ReportSummary,
)

# Reports are organization data, created ONLY by the research agent after the user approved the
# text (there is no POST /reports). Reading them is open to every role; creating and deleting
# needs admin or analyst. They are never cached.
MIN_REPORT_CHARS = 300
MAX_REPORT_CHARS = 20000
MAX_REPORT_TICKERS = 5
MAX_DATA_SOURCES = 30

# A citation marker is [1780] or a group [1780, 1781]. The number is a chunk_id until the report is
# saved, then the position 1..k
CITATION_MARKER = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


# The distinct numbers written in citation markers, in order of first appearance. Shared with the
# agent's answer citations (app/agent/run.py): the same marker rules must not exist twice
def find_marker_ids(text: str) -> list[int]:
    ids: dict[int, None] = {}
    for group in CITATION_MARKER.findall(text):
        for id_text in group.split(","):
            ids[int(id_text)] = None
    return list(ids)


# Renumbers the markers whose number is in valid_ids to 1..k in order of first appearance (a group
# is renumbered inside the brackets). Other markers stay as written. Returns the new text,
# {chunk_id: new number} and the ignored numbers (one entry per occurrence). Shared with
# app/agent/run.py
def renumber_markers(text: str, valid_ids: set[int]) -> tuple[str, dict[int, int], list[int]]:
    numbers: dict[int, int] = {}
    ignored: list[int] = []
    pieces = []
    position = 0
    for match in CITATION_MARKER.finditer(text):
        pieces.append(text[position : match.start()])
        parts = []
        for id_text in match.group(1).split(","):
            chunk_id = int(id_text)
            if chunk_id in valid_ids:
                if chunk_id not in numbers:
                    numbers[chunk_id] = len(numbers) + 1
                parts.append(str(numbers[chunk_id]))
            else:
                ignored.append(chunk_id)
                parts.append(id_text.strip())
        pieces.append("[" + ", ".join(parts) + "]")
        position = match.end()
    pieces.append(text[position:])
    return "".join(pieces), numbers, ignored


# Checks everything a report needs and returns what creating it needs: the cleaned title and
# content, the companies and the cited chunks as {chunk_id: (chunk, ticker)}. Writes nothing, so the
# agent calls it BEFORE it asks the user to approve (nobody approves a report that cannot be
# saved) and create_report calls it again after the approval (a role, a chunk or a company can
# change while an approval waits). found_chunks is {chunk_id: best similarity} of the
# search_filings results of the run. A report that cannot be saved is a ConflictError whose message
# tells the model what to fix
def check_can_create(
    db: Session,
    org_id: int,
    user_id: int,
    title: str,
    content: str,
    tickers: list[str],
    found_chunks: dict[int, float],
) -> tuple[str, str, list[Company], dict[int, tuple[DocumentChunk, str]]]:
    # 1. The role (read now, not from a token: a role can change while an approval waits)
    user = user_repository.get_by_id(db, org_id, user_id)
    if user is None or user.role not in (UserRole.admin, UserRole.analyst):
        raise ForbiddenError("Your role cannot save reports")

    # 2. Title and content
    title = title.strip()
    content = content.strip()
    if not 1 <= len(title) <= 200:
        raise ConflictError("The title must be 1 to 200 characters long")
    if len(content) < MIN_REPORT_CHARS:
        raise ConflictError(
            f"The report is too short: {len(content)} characters, at least {MIN_REPORT_CHARS} "
            "are needed. Write the full report."
        )
    if len(content) > MAX_REPORT_CHARS:
        raise ConflictError(
            f"The report is too long: {len(content)} characters, at most {MAX_REPORT_CHARS} "
            "are allowed. Shorten it."
        )

    # 3. Companies: distinct, in the RAG scope, stored
    allowed = [item.strip().upper() for item in settings.RAG_TICKERS.split(",") if item.strip()]
    distinct_tickers = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
    if not 1 <= len(distinct_tickers) <= MAX_REPORT_TICKERS:
        raise ConflictError(f"A report covers 1 to {MAX_REPORT_TICKERS} different companies")
    companies = []
    for ticker in distinct_tickers:
        if ticker not in allowed:
            raise ConflictError(
                f"Ticker {ticker} is not available. Available tickers: {', '.join(allowed)}"
            )
        company = company_repository.get_by_ticker(db, ticker)
        if company is None:
            raise NotFoundError(f"Company {ticker} not found")
        companies.append(company)

    # 4. Citations: at least one, and every marker must be the chunk_id of a passage that a search
    # of THIS run returned (unlike a chat answer, a flawed report is refused, not tolerated)
    marker_ids = find_marker_ids(content)
    if not marker_ids:
        raise ConflictError(
            "The report has no citations. Cite every statement taken from filing text with the "
            "chunk_id of a search_filings result of this run in square brackets, for example "
            "[1780]."
        )
    unknown_ids = [chunk_id for chunk_id in marker_ids if chunk_id not in found_chunks]
    if unknown_ids:
        raise ConflictError(
            f"These bracketed numbers are not chunk_ids returned by a search_filings call in this "
            f"run: {unknown_ids}. Do not use them. Cite only chunk_id values from search_filings "
            "results of this run, and do not write other numbers (such as years) in square "
            "brackets."
        )
    chunks = {
        chunk.id: (chunk, ticker) for chunk, ticker in chunk_repository.list_by_ids(db, marker_ids)
    }
    missing_ids = [chunk_id for chunk_id in marker_ids if chunk_id not in chunks]
    if missing_ids:
        raise ConflictError(
            f"These passages no longer exist: {missing_ids}. Search the filings again and cite "
            "the new chunk_ids."
        )
    return title, content, companies, chunks


def create_report(
    db: Session,
    org_id: int,
    user_id: int,
    run_id: int,
    title: str,
    content: str,
    tickers: list[str],
    found_chunks: dict[int, float],
    data_sources: list[dict],
) -> Report:
    title, content, companies, chunks = check_can_create(
        db, org_id, user_id, title, content, tickers, found_chunks
    )

    # Citations are renumbered 1..k by first appearance
    content, numbers, _ignored = renumber_markers(content, set(chunks))

    # The report belongs to the run only if that run is the caller's own (it may have been deleted
    # with its chat while the approval waited)
    run = agent_repository.get_run(db, org_id, user_id, run_id)
    report = report_repository.create(
        db,
        org_id,
        user_id,
        run.id if run is not None else None,
        title,
        content,
        data_sources[:MAX_DATA_SOURCES],
    )
    report_repository.add_companies(db, report.id, [company.id for company in companies])
    citations = []
    for chunk_id, number in numbers.items():
        chunk, chunk_ticker = chunks[chunk_id]
        citations.append(
            ReportCitation(
                report_id=report.id,
                number=number,
                chunk_id=chunk.id,
                score=found_chunks[chunk_id],
                filing_id=chunk.filing_id,
                ticker=chunk_ticker,
                fiscal_year=chunk.fiscal_year,
                section=chunk.section,
                content=chunk.content,
            )
        )
    report_repository.create_citations(db, citations)

    audit_repository.create(db, org_id, user_id, action="report.create", entity_id=report.id)
    db.commit()
    return report


def list_reports(db: Session, org_id: int, page: int, page_size: int) -> ReportListResponse:
    rows, total = report_repository.list_by_org(db, org_id, page_size, (page - 1) * page_size)
    tickers = report_repository.list_tickers(db, org_id, [report.id for report, _email in rows])
    return ReportListResponse(
        items=[
            ReportSummary(
                id=report.id,
                title=report.title,
                tickers=tickers[report.id],
                created_by=email,
                created_at=report.created_at,
            )
            for report, email in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


def get_report(db: Session, org_id: int, report_id: int) -> ReportDetail:
    row = report_repository.get_by_id(db, org_id, report_id)

    # Same message whether the report does not exist or belongs to another organization
    if row is None:
        raise NotFoundError("Report not found")
    report, email = row

    companies = report_repository.list_companies(db, org_id, report.id)
    citations = report_repository.list_citations(db, org_id, report.id)
    return ReportDetail(
        id=report.id,
        title=report.title,
        content=report.content,
        companies=[
            ReportCompanyResponse(ticker=company.ticker, name=company.name) for company in companies
        ],
        citations=[ReportCitationResponse.model_validate(citation) for citation in citations],
        data_sources=report.data_sources,
        created_by=email,
        agent_run_id=report.agent_run_id,
        created_at=report.created_at,
    )


def delete_report(db: Session, org_id: int, user_id: int, report_id: int) -> None:
    row = report_repository.get_by_id(db, org_id, report_id)
    if row is None:
        raise NotFoundError("Report not found")

    report_repository.delete(db, row[0])
    audit_repository.create(db, org_id, user_id, action="report.delete", entity_id=report_id)
    db.commit()
