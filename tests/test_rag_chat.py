import logging
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.exceptions import NotFoundError, ServiceUnavailableError
from app.models.audit import AuditLog
from app.models.chat import ChatMessage, ChatSession, Citation
from app.models.chunks import DocumentChunk
from app.models.users import User
from app.rag import chat, chunking, llm
from app.rag.llm import get_chat_model as real_get_chat_model  # the real one, not the test fake
from app.repositories import chat as chat_repository
from app.repositories import companies as company_repository
from app.repositories import filings as filing_repository
from app.repositories import organizations as organization_repository
from app.repositories import users as user_repository
from tests.conftest import ScriptedChatModel

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "filings"
EXPORT_QUESTION = "Export controls restrict sales of data center products to China."
MDNA_QUESTION = "Data Center revenue grew on demand for accelerated computing."
OFF_TOPIC = "What is a good recipe for banana bread?"


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Do not let a local .env change the numbers the tests expect
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", 0.30)
    monkeypatch.setattr(settings, "CHAT_HISTORY_MESSAGES", 6)
    monkeypatch.setattr(settings, "CHAT_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 5)
    monkeypatch.setattr(settings, "RETRIEVAL_CANDIDATES_K", 20)
    monkeypatch.setattr(settings, "RAG_TICKERS", "AAPL,NVDA")
    monkeypatch.setattr(settings, "RAG_LOOKBACK_YEARS", 2)
    # alembic's fileConfig can disable existing loggers in the test process (see Known issues)
    logging.getLogger("app.rag.chat").disabled = False


@pytest.fixture
def user(db: Session) -> User:
    organization = organization_repository.create(db, "Acme")
    return user_repository.create(db, organization.id, "viewer@acme.com", "hash", "viewer")


@pytest.fixture
def chat_session(db: Session, user: User) -> ChatSession:
    return chat.create_session(db, user.org_id, user.id)


def run(
    db: Session, user: User, chat_session: ChatSession, question: str, ticker: str | None = None
) -> list[dict]:
    return list(chat.ask(db, user.org_id, user.id, chat_session.id, question, ticker))


def messages_of(db: Session, chat_session: ChatSession) -> list[ChatMessage]:
    statement = select(ChatMessage).where(ChatMessage.session_id == chat_session.id)
    statement = statement.order_by(ChatMessage.id)
    return list(db.execute(statement).scalars().all())


def citations_of(db: Session, message_id: int) -> list[Citation]:
    statement = select(Citation).where(Citation.message_id == message_id).order_by(Citation.number)
    return list(db.execute(statement).scalars().all())


def count_rows(db: Session, model: type) -> int:
    return db.execute(select(func.count()).select_from(model)).scalar_one()


# ---------- the first question ----------


def test_first_question_streams_sources_tokens_done_and_saves_the_rows(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    answer = "Export controls limit data center sales to China [1]."
    model = script_chat(answer)
    target = chat_chunks["nvda_export"]

    events = run(db, user, chat_session, EXPORT_QUESTION, ticker="NVDA")

    # No history: no rewrite, so exactly one model call (the answer)
    assert len(model.received) == 1
    types = [event["type"] for event in events]
    assert types[0] == "sources"
    assert types[-1] == "done"
    assert set(types[1:-1]) == {"token"}
    assert "".join(event["text"] for event in events if event["type"] == "token") == answer

    sources = events[0]["sources"]
    assert len(sources) == 1
    assert sources[0]["number"] == 1
    assert sources[0]["chunk_id"] == target.id
    assert sources[0]["ticker"] == "NVDA"
    assert sources[0]["section"] == "risk_factors"
    assert sources[0]["fiscal_year"] == target.fiscal_year
    assert sources[0]["score"] == pytest.approx(1.0)
    assert sources[0]["content"] == target.content

    # The two messages and the one citation are saved
    user_message, assistant_message = messages_of(db, chat_session)
    assert (user_message.role, user_message.content) == ("user", EXPORT_QUESTION)
    assert user_message.rewritten_question is None
    assert user_message.ticker == "NVDA"
    assert (assistant_message.role, assistant_message.content) == ("assistant", answer)
    assert assistant_message.model == "gpt-5.4-mini"
    assert events[-1] == {
        "type": "done",
        "message_id": assistant_message.id,
        "cited_numbers": [1],
    }

    (citation,) = citations_of(db, assistant_message.id)
    assert citation.number == 1
    assert citation.chunk_id == target.id
    assert citation.score == pytest.approx(1.0)
    # The snapshot of what was cited
    assert citation.filing_id == target.filing_id
    assert citation.ticker == "NVDA"
    assert citation.fiscal_year == target.fiscal_year
    assert citation.section == "risk_factors"
    assert citation.content == target.content

    # The title comes from the first question
    db.refresh(chat_session)
    assert chat_session.title == EXPORT_QUESTION


@pytest.mark.parametrize(
    ("answer", "expected_numbers"),
    [
        ("Claim [1][1][3].", [1]),  # a repeat is stored once, [3] has no source
        ("Claim [1, 2] and more.", [1, 2]),  # the comma form
        ("Second [2] then first [1].", [1, 2]),  # stored in number order
        ("No markers at all.", []),
        ("Bad numbers [0] [3] [10] [-1] [a].", []),
    ],
)
def test_only_valid_distinct_numbers_become_citations(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected_numbers: list[int],
) -> None:
    # Everything passes the threshold and only 2 chunks are returned: exactly 2 sources
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", -1.0)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 2)
    script_chat(answer)

    events = run(db, user, chat_session, EXPORT_QUESTION)

    assert len(events[0]["sources"]) == 2
    assert events[-1]["cited_numbers"] == expected_numbers
    assistant_message = messages_of(db, chat_session)[1]
    stored = citations_of(db, assistant_message.id)
    assert [citation.number for citation in stored] == expected_numbers
    # Each stored citation matches the source with that number
    for citation in stored:
        source = events[0]["sources"][citation.number - 1]
        assert citation.chunk_id == source["chunk_id"]
        assert citation.content == source["content"]


def test_title_is_cut_to_100_characters_and_set_only_once(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    long_question = "  " + EXPORT_QUESTION + " " + "x" * 120 + "  "
    # First ask: answer. Second ask has history, so: rewrite, then answer
    script_chat("One [1].", EXPORT_QUESTION, "Two [1].")
    before = chat_session.updated_at

    run(db, user, chat_session, long_question)
    db.refresh(chat_session)
    assert chat_session.title == long_question.strip()[:100]
    assert len(chat_session.title) == 100
    assert chat_session.updated_at > before
    first_title = chat_session.title

    run(db, user, chat_session, "A different second question?")
    db.refresh(chat_session)
    assert chat_session.title == first_title


# ---------- follow-up questions ----------


def test_follow_up_makes_one_rewrite_call_and_retrieves_with_the_rewritten_question(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Call 1: answer to the first question. Call 2: the rewrite. Call 3: the answer to the rewrite
    model = script_chat("First answer [1].", MDNA_QUESTION, "Growth came from demand [1].")
    run(db, user, chat_session, EXPORT_QUESTION, ticker="NVDA")
    assert len(model.received) == 1

    # Spy on retrieve: it must receive the REWRITTEN question, and the ticker and scope filters
    seen = {}
    real_retrieve = chat.retrieve

    def spy(db, question, **kwargs):
        seen["question"] = question
        seen["kwargs"] = kwargs
        return real_retrieve(db, question, **kwargs)

    monkeypatch.setattr(chat, "retrieve", spy)

    events = run(db, user, chat_session, "And what about its growth?", ticker="NVDA")

    assert len(model.received) == 3  # exactly one more rewrite call and one more answer call
    assert seen["question"] == MDNA_QUESTION
    assert seen["kwargs"]["tickers"] == ["NVDA"]
    # The rewrite prompt saw the history and the follow-up
    assert EXPORT_QUESTION in model.prompt_text(1)
    assert "First answer [1]." in model.prompt_text(1)
    assert "And what about its growth?" in model.prompt_text(1)
    # The answer prompt uses the standalone question and the retrieved chunk
    assert f"Question: {MDNA_QUESTION}" in model.prompt_text(2)
    assert events[0]["sources"][0]["content"] == MDNA_QUESTION

    follow_up_message = messages_of(db, chat_session)[2]
    assert follow_up_message.content == "And what about its growth?"
    assert follow_up_message.rewritten_question == MDNA_QUESTION


def test_rewrite_only_sees_the_last_history_messages(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "CHAT_HISTORY_MESSAGES", 2)
    for number in range(1, 4):
        chat_repository.create_message(db, chat_session.id, "user", f"old question {number}")
        chat_repository.create_message(db, chat_session.id, "assistant", f"old answer {number}")
    db.commit()
    model = script_chat(MDNA_QUESTION, "Answer [1].")

    run(db, user, chat_session, "And then?")

    rewrite_prompt = model.prompt_text(0)
    assert "User: old question 3" in rewrite_prompt
    assert "Assistant: old answer 3" in rewrite_prompt
    assert "old question 2" not in rewrite_prompt
    assert "old answer 2" not in rewrite_prompt
    assert "old question 1" not in rewrite_prompt


def test_empty_rewrite_falls_back_to_the_original_question(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    chat_repository.create_message(db, chat_session.id, "user", "earlier")
    db.commit()
    script_chat("   ", "Answer [1].")

    events = run(db, user, chat_session, EXPORT_QUESTION)

    assert events[0]["sources"][0]["content"] == EXPORT_QUESTION
    follow_up = messages_of(db, chat_session)[1]
    assert follow_up.rewritten_question is None


# ---------- the relevance threshold ----------


def test_no_relevant_chunk_gives_the_fixed_answer_without_any_llm_call(
    db: Session, user: User, chat_session: ChatSession, chat_chunks: dict[str, DocumentChunk]
) -> None:
    # The default autouse fake FAILS the test if the model is called at all

    events = run(db, user, chat_session, OFF_TOPIC)

    assert events == [
        {"type": "sources", "sources": []},
        {"type": "token", "text": chat.NO_ANSWER},
        {"type": "done", "message_id": events[-1]["message_id"], "cited_numbers": []},
    ]
    assert chat.NO_ANSWER.startswith("I don't know.")
    user_message, assistant_message = messages_of(db, chat_session)
    assert assistant_message.content == chat.NO_ANSWER
    assert assistant_message.model is None
    assert events[-1]["message_id"] == assistant_message.id
    assert citations_of(db, assistant_message.id) == []
    assert user_message.content == OFF_TOPIC


def test_no_answer_call_for_a_follow_up_with_nothing_relevant(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    chat_repository.create_message(db, chat_session.id, "user", "earlier")
    db.commit()
    # Only the rewrite is scripted: a second call would raise StopIteration
    model = script_chat(OFF_TOPIC)

    events = run(db, user, chat_session, "and what about that?")

    assert len(model.received) == 1  # the rewrite happened, the answer call did not
    assert events[1] == {"type": "token", "text": chat.NO_ANSWER}
    follow_up = messages_of(db, chat_session)[1]
    assert follow_up.rewritten_question == OFF_TOPIC


def test_chunks_below_the_threshold_are_not_numbered_or_sent_to_the_model(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    model = script_chat("Answer [1].")

    # top_k 5 returns all 3 in-scope chunks, but only the one equal to the question passes
    events = run(db, user, chat_session, EXPORT_QUESTION)

    assert [source["content"] for source in events[0]["sources"]] == [EXPORT_QUESTION]
    prompt = model.prompt_text(0)
    year = chat_chunks["nvda_export"].fiscal_year
    assert f"[1] NVDA FY{year} 10-K, Risk Factors:\n{EXPORT_QUESTION}" in prompt
    assert "[2]" not in prompt
    assert chat_chunks["nvda_mdna"].content not in prompt
    assert chat_chunks["aapl_risk"].content not in prompt


# ---------- scope and filters ----------


def test_chunks_of_filings_outside_the_scope_are_never_sources(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The question equals the out-of-scope chunk: without the filter it would be source 1
    events = run(db, user, chat_session, chat_chunks["aapl_old"].content)
    assert events[0]["sources"] == []
    assert events[1]["text"] == chat.NO_ANSWER

    # Even when everything passes the threshold, the old chunk is not among the sources
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", -1.0)
    monkeypatch.setattr(settings, "RETRIEVAL_TOP_K", 10)
    # The first run saved messages, so this one has history: a rewrite call, then the answer
    script_chat(EXPORT_QUESTION, "Answer.")
    events = run(db, user, chat_session, "And the other one?")
    chunk_ids = {source["chunk_id"] for source in events[0]["sources"]}
    assert chunk_ids == {
        chat_chunks["nvda_export"].id,
        chat_chunks["nvda_mdna"].id,
        chat_chunks["aapl_risk"].id,
    }


def test_an_empty_scope_answers_without_searching(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An empty filing_ids list would mean "no filter" in retrieve, so chat must not call it
    monkeypatch.setattr(chat, "list_scope_filing_ids", lambda db: [])

    def must_not_search(*args, **kwargs):
        raise AssertionError("retrieve was called with an empty scope")

    monkeypatch.setattr(chat, "retrieve", must_not_search)

    events = run(db, user, chat_session, EXPORT_QUESTION)

    assert events[0] == {"type": "sources", "sources": []}
    assert events[1] == {"type": "token", "text": chat.NO_ANSWER}


def test_ticker_filter_limits_the_sources(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "RELEVANCE_THRESHOLD", -1.0)
    script_chat("Answer.")

    events = run(db, user, chat_session, EXPORT_QUESTION, ticker="AAPL")

    assert [source["ticker"] for source in events[0]["sources"]] == ["AAPL"]
    assert messages_of(db, chat_session)[0].ticker == "AAPL"


# ---------- errors after the stream started ----------


def test_a_mid_stream_failure_sends_an_error_event_and_saves_nothing(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
    caplog: pytest.LogCaptureFixture,
) -> None:
    script_chat("one two three four five [1]", fail_after_pieces=2)

    with caplog.at_level(logging.ERROR, logger="app.rag.chat"):
        events = run(db, user, chat_session, EXPORT_QUESTION)

    types = [event["type"] for event in events]
    assert types == ["sources", "token", "token", "error"]
    assert events[-1]["detail"] == "The answer could not be completed. Try again."
    # Nothing saved, and the title was not set
    assert messages_of(db, chat_session) == []
    assert count_rows(db, Citation) == 0
    db.refresh(chat_session)
    assert chat_session.title is None
    # Only the class name is logged, never the exception message
    assert "APIConnectionError" in caplog.text
    assert "example.invalid" not in caplog.text


def test_a_failing_rewrite_sends_an_error_event_and_saves_nothing(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    chat_repository.create_message(db, chat_session.id, "user", "earlier")
    db.commit()  # in production an earlier request committed it
    model = script_chat()
    model.fail_on_call = True

    events = run(db, user, chat_session, "and then?")

    assert [event["type"] for event in events] == ["error"]
    assert len(messages_of(db, chat_session)) == 1  # only the message created by the test


def test_an_empty_model_answer_is_an_error_and_saves_nothing(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat("")

    events = run(db, user, chat_session, EXPORT_QUESTION)

    assert [event["type"] for event in events] == ["sources", "error"]
    assert messages_of(db, chat_session) == []


# ---------- checks before the stream ----------


def test_unknown_session_raises_before_any_event(
    db: Session, user: User, chat_chunks: dict[str, DocumentChunk]
) -> None:
    # ask() checks eagerly: the exception is raised by the call itself, not on the first event
    with pytest.raises(NotFoundError, match="Chat session not found"):
        chat.ask(db, user.org_id, user.id, 999999, "question", None)


def test_missing_key_raises_before_any_event(
    db: Session, user: User, chat_session: ChatSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_key() -> None:
        raise ValueError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(llm, "get_chat_model", no_key)

    with pytest.raises(ServiceUnavailableError, match="Chat is disabled: OPENAI_API_KEY not set"):
        chat.ask(db, user.org_id, user.id, chat_session.id, "question", None)


def test_get_chat_model_needs_a_key_and_uses_the_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        real_get_chat_model()

    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(settings, "CHAT_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 42)
    model = real_get_chat_model()  # creating the object makes no network call
    assert model.model_name == "gpt-5.4-mini"
    assert model.request_timeout == 42
    assert model.max_retries == 2
    assert model.temperature is None


# ---------- prompts ----------


def test_qa_prompt_has_the_rules_and_the_numbered_excerpts() -> None:
    excerpts = (
        "[1] NVDA FY2026 10-K, Risk Factors:\nFirst text\n\n[2] AAPL FY2025 10-K, MD&A:\nSecond"
    )

    system_message, human_message = chat.QA_PROMPT.format_messages(
        excerpts=excerpts, question="What changed?"
    )

    assert system_message.type == "system"
    for rule in ("ONLY", "[1][3]", "don't know", "outside knowledge", "investment advice"):
        assert rule in system_message.content
    assert "[1] NVDA FY2026 10-K, Risk Factors:\nFirst text" in human_message.content
    assert "[2] AAPL FY2025 10-K, MD&A:\nSecond" in human_message.content
    assert human_message.content.endswith("Question: What changed?")


# ---------- sessions: audit, cascade, citations after a re-embed ----------


def test_create_and_delete_write_audit_rows_and_messages_write_none(
    db: Session,
    user: User,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat("Answer [1].")
    created = chat.create_session(db, user.org_id, user.id)
    run(db, user, created, EXPORT_QUESTION)
    chat.delete_session(db, user.org_id, user.id, created.id)

    statement = select(AuditLog.action, AuditLog.entity_id).where(AuditLog.org_id == user.org_id)
    rows = sorted(db.execute(statement).all())
    assert rows == [("chat_session.create", created.id), ("chat_session.delete", created.id)]


def test_deleting_a_session_deletes_its_messages_and_citations(
    db: Session,
    user: User,
    chat_session: ChatSession,
    chat_chunks: dict[str, DocumentChunk],
    script_chat: Callable[..., ScriptedChatModel],
) -> None:
    script_chat("Answer [1].")
    run(db, user, chat_session, EXPORT_QUESTION)
    assert count_rows(db, ChatMessage) == 2
    assert count_rows(db, Citation) == 1

    chat.delete_session(db, user.org_id, user.id, chat_session.id)

    assert count_rows(db, ChatSession) == 0
    assert count_rows(db, ChatMessage) == 0
    assert count_rows(db, Citation) == 0
    with pytest.raises(NotFoundError):
        chat.get_session_detail(db, user.org_id, user.id, chat_session.id)


def test_sessions_list_is_newest_first_and_personal(db: Session, user: User) -> None:
    other_user = user_repository.create(db, user.org_id, "other@acme.com", "hash", "analyst")
    first = chat.create_session(db, user.org_id, user.id)
    second = chat.create_session(db, user.org_id, user.id)
    chat.create_session(db, user.org_id, other_user.id)
    chat_repository.touch_session(db, first)  # now the most recently used

    sessions = chat.list_sessions(db, user.org_id, user.id)

    assert [s.id for s in sessions] == [first.id, second.id]


def test_citation_survives_a_reembed_with_its_snapshot(
    db: Session,
    user: User,
    chat_session: ChatSession,
    script_chat: Callable[..., ScriptedChatModel],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A real filing (the AAPL fixture) chunked by the real job, so --reembed can run on it
    monkeypatch.setattr(settings, "RAW_DATA_DIR", str(FIXTURES_DIR))
    monkeypatch.setattr(settings, "CHUNK_SIZE", 300)
    monkeypatch.setattr(settings, "CHUNK_OVERLAP", 50)
    company = company_repository.create(db, "AAPL", "0000320193", "AAPL Inc.", None)
    today = date.today()
    filing = filing_repository.create(
        db,
        company.id,
        "0000000000-26-000001",
        "10-K",
        today - timedelta(days=60),
        report_date=today - timedelta(days=90),
        fiscal_year=today.year - 1,
        primary_document="document.htm",
    )
    filing.raw_path = "aapl_10k_excerpt.htm"
    db.flush()
    chunking.embed_filings(db, filing_ids=[filing.id])
    first_chunk = (
        db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.filing_id == filing.id, DocumentChunk.section == "risk_factors")
            .order_by(DocumentChunk.chunk_index)
        )
        .scalars()
        .first()
    )
    cited_text = first_chunk.content
    cited_chunk_id = first_chunk.id
    script_chat("Answer [1].")
    run(db, user, chat_session, cited_text)
    (citation,) = db.execute(select(Citation)).scalars().all()
    assert citation.chunk_id == cited_chunk_id

    chunking.embed_filings(db, replace=True, filing_ids=[filing.id])

    # The old chunk row is gone, the citation stays: chunk_id is null, the snapshot is intact
    assert db.get(DocumentChunk, cited_chunk_id) is None
    db.refresh(citation)
    assert citation.chunk_id is None
    assert citation.content == cited_text
    detail = chat.get_session_detail(db, user.org_id, user.id, chat_session.id)
    (cited,) = detail.messages[1].citations
    assert cited.chunk_id is None
    assert cited.content == cited_text
    assert cited.ticker == "AAPL"
    assert cited.section == "risk_factors"
