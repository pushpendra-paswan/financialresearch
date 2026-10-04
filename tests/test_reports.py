import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.exceptions import ConflictError, ForbiddenError, NotFoundError
from app.models.audit import AuditLog
from app.models.chunks import DocumentChunk
from app.models.reports import Report, ReportCitation, ReportCompany
from app.rag import chat
from app.repositories import agent as agent_repository
from app.repositories import chat as chat_repository
from app.repositories import reports as report_repository
from app.services.reports import (
    MAX_DATA_SOURCES,
    MAX_REPORT_CHARS,
    MIN_REPORT_CHARS,
    check_can_create,
    create_report,
    find_marker_ids,
    renumber_markers,
)

FILLER = "This sentence only makes the report long enough. " * 8  # about 390 characters


@pytest.fixture(autouse=True)
def report_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA")


@pytest.fixture
def chunks(chat_chunks: dict[str, DocumentChunk]) -> dict[str, DocumentChunk]:
    return chat_chunks


def content_with(*chunk_ids: int, text: str = FILLER) -> str:
    # A report text citing the given chunk ids, one sentence each
    sentences = " ".join(
        f"Statement number {i} from the filings [{cid}]." for i, cid in enumerate(chunk_ids)
    )
    return f"## Summary\n\n{sentences}\n\n{text}"


def save(
    db: Session,
    person: dict,
    chunks: dict[str, DocumentChunk],
    title: str = "Apple vs NVIDIA",
    tickers: tuple[str, ...] = ("AAPL", "NVDA"),
    run_id: int = 0,
    data_sources: list[dict] | None = None,
) -> Report:
    ids = [chunks["aapl_risk"].id, chunks["nvda_export"].id]
    return create_report(
        db,
        person["org_id"],
        person["user_id"],
        run_id,
        title,
        content_with(*ids),
        list(tickers),
        {chunk_id: 0.9 for chunk_id in ids},
        data_sources or [],
    )


def check(db: Session, person: dict, **changes: object) -> tuple:
    # check_can_create with a valid report, changed by the keyword arguments
    ids = changes.pop("ids", None)
    found = changes.pop("found_chunks", None)
    arguments = {
        "title": "Title",
        "content": content_with(*(ids or [])),
        "tickers": ["AAPL"],
        "found_chunks": found,
    }
    arguments.update(changes)
    return check_can_create(db, person["org_id"], person["user_id"], **arguments)


def make_run(db: Session, person: dict) -> int:
    session = chat.create_session(db, person["org_id"], person["user_id"])
    message = chat_repository.create_message(db, session.id, "user", "question")
    run = agent_repository.create_run(
        db, person["org_id"], person["user_id"], session.id, message.id
    )
    return run.id


def count(db: Session, model: type) -> int:
    return db.execute(select(func.count()).select_from(model)).scalar_one()


# ---------- list and detail ----------


def test_detail_has_companies_citations_and_data_sources(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    sources = [{"tool": "get_financials", "args": {"ticker": "AAPL"}}]
    report = save(db, people["analyst"], chunks, data_sources=sources)

    body = client.get(f"/reports/{report.id}", headers=people["viewer"]["headers"]).json()

    assert body["id"] == report.id
    assert body["title"] == "Apple vs NVIDIA"
    assert body["created_by"] == "analyst@acme.com"
    assert body["agent_run_id"] is None
    assert [c["ticker"] for c in body["companies"]] == ["AAPL", "NVDA"]
    assert body["companies"][0]["name"] == "Apple Inc."
    assert body["data_sources"] == sources
    # Renumbered by first appearance, with the stored snapshot
    assert "[1]" in body["content"] and "[2]" in body["content"]
    first, second = body["citations"]
    assert (first["number"], first["chunk_id"], first["ticker"]) == (
        1,
        chunks["aapl_risk"].id,
        "AAPL",
    )
    assert (second["number"], second["ticker"], second["section"]) == (2, "NVDA", "risk_factors")
    assert first["content"] == chunks["aapl_risk"].content
    assert first["score"] == pytest.approx(0.9)
    assert set(first) == {
        "number",
        "chunk_id",
        "ticker",
        "fiscal_year",
        "section",
        "score",
        "content",
    }


def test_list_is_newest_first_with_tickers_creator_and_pagination(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    saved = [
        save(db, people["analyst"], chunks, title="First", tickers=("AAPL",)),
        save(db, people["admin"], chunks, title="Second", tickers=("NVDA", "AAPL")),
        save(db, people["analyst"], chunks, title="Third", tickers=("NVDA",)),
    ]
    headers = people["viewer"]["headers"]

    body = client.get("/reports", headers=headers).json()

    assert body["total"] == 3 and body["page"] == 1 and body["page_size"] == 20
    assert [item["title"] for item in body["items"]] == ["Third", "Second", "First"]
    assert [item["id"] for item in body["items"]] == [r.id for r in reversed(saved)]
    assert [item["tickers"] for item in body["items"]] == [["NVDA"], ["AAPL", "NVDA"], ["AAPL"]]
    assert [item["created_by"] for item in body["items"]] == [
        "analyst@acme.com",
        "admin@acme.com",
        "analyst@acme.com",
    ]
    assert set(body["items"][0]) == {"id", "title", "tickers", "created_by", "created_at"}

    page_two = client.get("/reports?page=2&page_size=2", headers=headers).json()
    assert page_two["total"] == 3
    assert [item["title"] for item in page_two["items"]] == ["First"]


def test_list_validates_the_paging_arguments(client: TestClient, people: dict) -> None:
    headers = people["viewer"]["headers"]
    assert client.get("/reports?page=0", headers=headers).status_code == 422
    assert client.get("/reports?page_size=101", headers=headers).status_code == 422
    assert client.get("/reports?page_size=0", headers=headers).status_code == 422


def test_every_role_can_list_and_read(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    report = save(db, people["analyst"], chunks)
    for role in ("admin", "analyst", "viewer", "colleague"):
        headers = people[role]["headers"]
        assert client.get("/reports", headers=headers).json()["total"] == 1
        assert client.get(f"/reports/{report.id}", headers=headers).status_code == 200


def test_reports_need_a_login(client: TestClient) -> None:
    assert client.get("/reports").status_code == 401
    assert client.get("/reports/1").status_code == 401
    assert client.delete("/reports/1").status_code == 401


# ---------- organizations ----------


def test_another_organization_sees_and_changes_nothing(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    report = save(db, people["analyst"], chunks)
    outsider = people["outsider"]["headers"]

    assert client.get("/reports", headers=outsider).json()["total"] == 0
    other = client.get(f"/reports/{report.id}", headers=outsider)
    missing = client.get("/reports/999999", headers=outsider)
    assert (other.status_code, other.json()) == (404, {"detail": "Report not found"})
    assert (missing.status_code, missing.json()) == (404, other.json())

    deleted = client.delete(f"/reports/{report.id}", headers=outsider)
    assert (deleted.status_code, deleted.json()) == (404, {"detail": "Report not found"})
    assert count(db, Report) == 1
    assert count(db, ReportCompany) == 2
    assert count(db, ReportCitation) == 2


def test_every_repository_read_takes_the_organization(
    db: Session, people: dict, chunks: dict
) -> None:
    report = save(db, people["analyst"], chunks)
    other_org = people["outsider"]["org_id"]
    assert report_repository.get_by_id(db, other_org, report.id) is None
    assert report_repository.list_by_org(db, other_org, 20, 0) == ([], 0)
    assert report_repository.list_companies(db, other_org, report.id) == []
    assert report_repository.list_citations(db, other_org, report.id) == []
    assert report_repository.list_tickers(db, other_org, [report.id]) == {report.id: []}


# ---------- delete ----------


def test_a_viewer_cannot_delete_and_gets_403_before_any_lookup(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    report = save(db, people["analyst"], chunks)
    headers = people["viewer"]["headers"]

    existing = client.delete(f"/reports/{report.id}", headers=headers)
    missing = client.delete("/reports/999999", headers=headers)

    assert existing.status_code == 403 and missing.status_code == 403
    assert existing.json() == missing.json() == {"detail": "Admin or analyst role required"}
    assert count(db, Report) == 1


def test_an_editor_deletes_the_report_with_its_companies_and_citations(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    keep = save(db, people["analyst"], chunks, title="Keep")
    gone = save(db, people["analyst"], chunks, title="Gone")

    response = client.delete(f"/reports/{gone.id}", headers=people["colleague"]["headers"])

    assert response.status_code == 204
    db.expire_all()
    assert [r.id for r in db.execute(select(Report)).scalars()] == [keep.id]
    # The database cascade removed the rows of the deleted report only
    assert count(db, ReportCompany) == 2
    assert count(db, ReportCitation) == 2
    assert client.get(f"/reports/{gone.id}", headers=people["viewer"]["headers"]).status_code == 404
    # The chunks and the catalog are untouched
    assert count(db, DocumentChunk) == 4


def test_audit_rows_for_create_and_delete(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    report = save(db, people["analyst"], chunks)
    client.delete(f"/reports/{report.id}", headers=people["admin"]["headers"])

    rows = db.execute(
        select(AuditLog.action, AuditLog.entity_id, AuditLog.user_id)
        .where(AuditLog.action.like("report.%"))
        .order_by(AuditLog.id)
    ).all()
    assert rows == [
        ("report.create", report.id, people["analyst"]["user_id"]),
        ("report.delete", report.id, people["admin"]["user_id"]),
    ]


def test_no_endpoint_creates_a_report(client: TestClient, people: dict) -> None:
    for role in ("admin", "analyst"):
        response = client.post(
            "/reports",
            json={"title": "x", "content": "y", "tickers": ["AAPL"]},
            headers=people[role]["headers"],
        )
        assert response.status_code == 405
    # The API describes only reads and the delete
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths["/reports"]) == {"get"}
    assert set(paths["/reports/{report_id}"]) == {"get", "delete"}


# ---------- snapshots and links ----------


def test_the_citation_snapshot_stays_readable_after_the_chunk_row_is_deleted(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    report = save(db, people["analyst"], chunks)
    text = chunks["aapl_risk"].content
    db.delete(chunks["aapl_risk"])
    db.flush()
    db.expire_all()

    body = client.get(f"/reports/{report.id}", headers=people["viewer"]["headers"]).json()

    first = body["citations"][0]
    assert first["chunk_id"] is None
    assert first["content"] == text and first["ticker"] == "AAPL"


def test_deleting_the_chat_leaves_the_report_without_a_run(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    analyst = people["analyst"]
    session = chat.create_session(db, analyst["org_id"], analyst["user_id"])
    message = chat_repository.create_message(db, session.id, "user", "question")
    run = agent_repository.create_run(
        db, analyst["org_id"], analyst["user_id"], session.id, message.id
    )
    report = save(db, analyst, chunks, run_id=run.id)
    assert report.agent_run_id == run.id

    chat.delete_session(db, analyst["org_id"], analyst["user_id"], session.id)

    db.expire_all()
    body = client.get(f"/reports/{report.id}", headers=analyst["headers"]).json()
    assert body["agent_run_id"] is None
    assert len(body["citations"]) == 2


def test_the_run_is_linked_only_when_it_belongs_to_the_same_organization_and_user(
    db: Session, people: dict, chunks: dict
) -> None:
    own_run = make_run(db, people["analyst"])
    colleague_run = make_run(db, people["colleague"])
    outsider_run = make_run(db, people["outsider"])

    linked = save(db, people["analyst"], chunks, run_id=own_run)
    colleagues = save(db, people["analyst"], chunks, run_id=colleague_run)
    foreign = save(db, people["analyst"], chunks, run_id=outsider_run)
    unknown = save(db, people["analyst"], chunks, run_id=999999)

    assert linked.agent_run_id == own_run
    assert colleagues.agent_run_id is None
    assert foreign.agent_run_id is None
    assert unknown.agent_run_id is None


def test_data_sources_are_capped(db: Session, people: dict, chunks: dict) -> None:
    sources = [
        {"tool": "get_financials", "args": {"years": n}} for n in range(MAX_DATA_SOURCES + 5)
    ]
    report = save(db, people["analyst"], chunks, data_sources=sources)
    assert report.data_sources == sources[:MAX_DATA_SOURCES]


# ---------- validation ----------


def test_a_viewer_and_a_missing_user_are_refused(db: Session, people: dict, chunks: dict) -> None:
    ids = [chunks["aapl_risk"].id]
    found = {ids[0]: 0.9}
    with pytest.raises(ForbiddenError, match="Your role cannot save reports"):
        check(db, people["viewer"], ids=ids, found_chunks=found)
    # A user of another organization is not found in this one
    with pytest.raises(ForbiddenError, match="Your role cannot save reports"):
        check_can_create(
            db,
            people["outsider"]["org_id"],
            people["analyst"]["user_id"],
            "T",
            content_with(*ids),
            ["AAPL"],
            found,
        )
    # Admin and analyst pass
    for role in ("admin", "analyst"):
        check(db, people[role], ids=ids, found_chunks=found)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"title": ""}, "The title must be 1 to 200 characters long"),
        ({"title": "   "}, "The title must be 1 to 200 characters long"),
        ({"title": "x" * 201}, "The title must be 1 to 200 characters long"),
        ({"tickers": []}, "A report covers 1 to 5 different companies"),
        (
            {"tickers": ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META"]},
            "A report covers 1 to 5 different companies",
        ),
        ({"tickers": ["MSFT"]}, "Ticker MSFT is not available. Available tickers: AAPL, NVDA"),
    ],
)
def test_title_and_ticker_limits(
    db: Session, people: dict, chunks: dict, changes: dict, message: str
) -> None:
    ids = [chunks["aapl_risk"].id]
    with pytest.raises(ConflictError, match=message):
        check(db, people["analyst"], ids=ids, found_chunks={ids[0]: 0.9}, **changes)


def test_the_limits_are_inclusive(db: Session, people: dict, chunks: dict) -> None:
    ids = [chunks["aapl_risk"].id]
    found = {ids[0]: 0.9}
    check(db, people["analyst"], ids=ids, found_chunks=found, title="x" * 200)
    # Duplicates count once: 6 names, 2 different companies
    check(
        db,
        people["analyst"],
        ids=ids,
        found_chunks=found,
        tickers=["aapl", "AAPL", " nvda ", "NVDA", "Aapl", "nvda"],
    )
    exact_min = content_with(*ids)
    exact_min = exact_min[:-1] if False else exact_min
    padded = f"[{ids[0]}] " + "x" * (MIN_REPORT_CHARS - len(f"[{ids[0]}] "))
    assert len(padded) == MIN_REPORT_CHARS
    check(db, people["analyst"], ids=ids, found_chunks=found, content=padded)
    biggest = f"[{ids[0]}] " + "x" * (MAX_REPORT_CHARS - len(f"[{ids[0]}] "))
    assert len(biggest) == MAX_REPORT_CHARS
    check(db, people["analyst"], ids=ids, found_chunks=found, content=biggest)


def test_content_length_limits(db: Session, people: dict, chunks: dict) -> None:
    ids = [chunks["aapl_risk"].id]
    found = {ids[0]: 0.9}
    short = f"[{ids[0]}] " + "x" * (MIN_REPORT_CHARS - 1 - len(f"[{ids[0]}] "))
    with pytest.raises(ConflictError, match="too short: 299 characters, at least 300"):
        check(db, people["analyst"], ids=ids, found_chunks=found, content=short)
    long = f"[{ids[0]}] " + "x" * (MAX_REPORT_CHARS + 1 - len(f"[{ids[0]}] "))
    with pytest.raises(ConflictError, match="too long: 20001 characters, at most 20000"):
        check(db, people["analyst"], ids=ids, found_chunks=found, content=long)


def test_a_ticker_without_a_company_row_is_not_found(
    db: Session, people: dict, chunks: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In the RAG scope but not stored in the catalog
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA,ZZZZ")
    ids = [chunks["aapl_risk"].id]
    with pytest.raises(NotFoundError, match="Company ZZZZ not found"):
        check(db, people["analyst"], ids=ids, found_chunks={ids[0]: 0.9}, tickers=["ZZZZ"])


def test_a_report_without_markers_is_refused_with_advice(db: Session, people: dict) -> None:
    with pytest.raises(ConflictError) as error:
        check(db, people["analyst"], ids=[], found_chunks={})
    assert "no citations" in error.value.message and "chunk_id" in error.value.message


def test_markers_that_no_search_of_the_run_returned_are_refused_and_listed(
    db: Session, people: dict, chunks: dict
) -> None:
    good = chunks["aapl_risk"].id
    other = chunks["nvda_export"].id  # exists, but the run never searched it
    content = f"Fine [{good}]. Bad [{other}]. A year [2025]. " + FILLER

    with pytest.raises(ConflictError) as error:
        check(db, people["analyst"], content=content, found_chunks={good: 0.9})

    assert f"[{other}, 2025]" in error.value.message
    assert str(good) not in error.value.message
    assert "years" in error.value.message


def test_a_group_marker_is_checked_id_by_id(db: Session, people: dict, chunks: dict) -> None:
    good = chunks["aapl_risk"].id
    other = chunks["nvda_export"].id
    content = f"Both [{good}, {other}]. " + FILLER

    with pytest.raises(ConflictError) as error:
        check(db, people["analyst"], content=content, found_chunks={good: 0.9})
    assert f"[{other}]" in error.value.message

    # Both returned by searches: fine
    check(db, people["analyst"], content=content, found_chunks={good: 0.9, other: 0.8})


def test_a_chunk_row_that_no_longer_exists_is_refused(
    db: Session, people: dict, chunks: dict
) -> None:
    gone = chunks["aapl_risk"].id
    db.delete(chunks["aapl_risk"])
    db.flush()

    with pytest.raises(ConflictError) as error:
        check(db, people["analyst"], ids=[gone], found_chunks={gone: 0.9})
    assert f"[{gone}]" in error.value.message and "no longer exist" in error.value.message


def test_create_report_checks_again_and_writes_nothing_when_it_fails(
    db: Session, people: dict, chunks: dict
) -> None:
    ids = [chunks["aapl_risk"].id]
    with pytest.raises(ForbiddenError):
        create_report(
            db,
            people["viewer"]["org_id"],
            people["viewer"]["user_id"],
            0,
            "T",
            content_with(*ids),
            ["AAPL"],
            {ids[0]: 0.9},
            [],
        )
    assert count(db, Report) == 0 and count(db, ReportCompany) == 0
    assert count(db, ReportCitation) == 0


def test_the_cleaned_title_and_content_are_stored(db: Session, people: dict, chunks: dict) -> None:
    ids = [chunks["aapl_risk"].id]
    report = create_report(
        db,
        people["analyst"]["org_id"],
        people["analyst"]["user_id"],
        0,
        "  Padded title  ",
        "\n\n" + content_with(*ids) + "\n\n",
        [" aapl "],
        {ids[0]: 0.9},
        [],
    )
    assert report.title == "Padded title"
    assert report.content == report.content.strip()
    assert report.org_id == people["analyst"]["org_id"]
    assert report.user_id == people["analyst"]["user_id"]


# ---------- renumbering ----------


def test_find_marker_ids_lists_distinct_numbers_in_order() -> None:
    assert find_marker_ids("a [30] b [2, 30] c [4,5] d [x] e [] f [1 2]") == [30, 2, 4, 5]
    assert find_marker_ids("no markers") == []


def test_markers_become_one_to_k_by_first_appearance() -> None:
    text, numbers, ignored = renumber_markers("A [50]. B [20]. C [50]. D [7].", {50, 20, 7})
    assert text == "A [1]. B [2]. C [1]. D [3]."
    assert numbers == {50: 1, 20: 2, 7: 3}
    assert ignored == []


def test_a_group_is_renumbered_inside_the_brackets_and_other_markers_stay() -> None:
    text, numbers, ignored = renumber_markers("A [50, 20] B [20,50] C [999] D [2025, 20]", {50, 20})
    assert text == "A [1, 2] B [2, 1] C [999] D [2025, 2]"
    assert numbers == {50: 1, 20: 2}
    assert ignored == [999, 2025]


def test_create_report_renumbers_and_stores_snapshot_and_score(
    db: Session, people: dict, chunks: dict
) -> None:
    apple, nvda = chunks["aapl_risk"], chunks["nvda_export"]
    content = (
        f"## Risks\n\nNVIDIA first [{nvda.id}]. Apple next [{apple.id}]. Again NVIDIA "
        f"[{nvda.id}]. Both [{apple.id}, {nvda.id}].\n\n" + FILLER
    )

    report = create_report(
        db,
        people["analyst"]["org_id"],
        people["analyst"]["user_id"],
        0,
        "Risks",
        content,
        ["NVDA", "AAPL"],
        {nvda.id: 0.81, apple.id: 0.64},
        [],
    )

    assert report.content.startswith(
        "## Risks\n\nNVIDIA first [1]. Apple next [2]. Again NVIDIA [1]. Both [2, 1]."
    )
    rows = (
        db.execute(
            select(ReportCitation).where(ReportCitation.report_id == report.id).order_by("number")
        )
        .scalars()
        .all()
    )
    # One row per distinct chunk, even though [1] appears twice
    assert [(r.number, r.chunk_id, r.ticker, r.section) for r in rows] == [
        (1, nvda.id, "NVDA", "risk_factors"),
        (2, apple.id, "AAPL", "risk_factors"),
    ]
    assert [r.score for r in rows] == pytest.approx([0.81, 0.64])
    assert rows[0].content == nvda.content and rows[1].content == apple.content
    assert rows[0].filing_id == nvda.filing_id
    assert rows[0].fiscal_year == nvda.fiscal_year
    # The companies, sorted by ticker
    companies = report_repository.list_companies(db, people["analyst"]["org_id"], report.id)
    assert [company.ticker for company in companies] == ["AAPL", "NVDA"]


def test_the_list_runs_a_fixed_number_of_queries(
    client: TestClient, db: Session, people: dict, chunks: dict
) -> None:
    from sqlalchemy import event

    for number in range(5):
        save(db, people["analyst"], chunks, title=f"Report {number}")
    statements: list[str] = []

    def record(conn, cursor, statement, *args):  # noqa: ANN001, ANN002
        statements.append(statement)

    engine = db.get_bind()
    event.listen(engine, "before_cursor_execute", record)
    try:
        body = client.get("/reports", headers=people["viewer"]["headers"]).json()
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert len(body["items"]) == 5
    reports_queries = [s for s in statements if "FROM reports" in s or "report_companies" in s]
    # count, page, tickers: 3 queries for 5 reports (no query per row)
    assert len(reports_queries) == 3
