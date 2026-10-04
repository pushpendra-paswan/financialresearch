import logging
import re
from collections.abc import Iterator

import openai
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.exceptions import NotFoundError, ServiceUnavailableError
from app.models.chat import ChatSession, Citation
from app.rag import llm
from app.rag.parsing import list_scope_filing_ids
from app.rag.retrieval import retrieve
from app.repositories import agent as agent_repository
from app.repositories import audit as audit_repository
from app.repositories import chat as chat_repository
from app.schemas.chat import CitationResponse, MessageResponse, SessionDetailResponse

logger = logging.getLogger(__name__)

# Returned when no retrieved chunk is relevant. The answer model is NOT called in that case
NO_ANSWER = (
    "I don't know. The available filings do not contain information relevant to this question."
)

# Short names of the sections for the excerpt labels (the keys of SECTIONS in parsing.py)
SECTION_NAMES = {"risk_factors": "Risk Factors", "mdna": "MD&A"}

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite a follow-up question into a standalone question, using the chat history "
            "only to resolve references such as 'it', 'that' or 'the company'. Keep the meaning "
            "and keep company names and years. If the question is already standalone, return it "
            "unchanged. Do NOT answer the question. Reply with the rewritten question only.",
        ),
        (
            "human",
            "Chat history:\n{history}\n\nFollow-up question: {question}\n\nStandalone question:",
        ),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the 10-K filings of public companies. Rules:\n"
            "- Answer ONLY from the numbered excerpts below. Use no outside knowledge.\n"
            "- Cite every factual claim with the number of the excerpt that supports it, in "
            "square brackets, one number per bracket, for example [1][3]. Put the citation right "
            "after the sentence it supports: every sentence that states a fact needs its own "
            "citation, not one at the end of the answer. Never cite a number that is not in the "
            "list.\n"
            "- If the excerpts do not contain the answer, say that you don't know.\n"
            "- Do not give investment advice and do not predict prices.",
        ),
        ("human", "Excerpts:\n\n{excerpts}\n\nQuestion: {question}"),
    ]
)


# Steps 4 and 6 of the chat flow, shared by `ask` and the evaluation script (evals/run_eval.py),
# so the evaluation measures exactly what chat does. Retrieves inside the RAG scope, keeps the
# chunks that pass the relevance threshold, numbers them and builds the excerpt text for the
# prompt. Returns (retrieved, passing, sources, context_text): all retrieved documents, the ones
# that passed, the numbered sources for the client, and the labelled excerpts ("" when none passed)
def prepare_context(
    db: Session, question: str, ticker: str | None
) -> tuple[list[Document], list[Document], list[dict], str]:
    # Retrieve, restricted to the exact RAG scope. An EMPTY filing_ids list would mean "no
    # filter" in retrieve, so an empty scope must skip the search instead
    scope_filing_ids = list_scope_filing_ids(db)
    retrieved = []
    if scope_filing_ids:
        retrieved = retrieve(
            db,
            question,
            tickers=[ticker] if ticker else None,
            filing_ids=scope_filing_ids,
        )
    # Keep only relevant chunks. The threshold uses vector_similarity, not the RRF score or the
    # rerank score (those only order chunks). The order from retrieve is kept
    passing = [
        document
        for document in retrieved
        if document.metadata["vector_similarity"] >= settings.RELEVANCE_THRESHOLD
    ]

    sources = [
        {
            "number": number,
            "chunk_id": document.metadata["chunk_id"],
            "ticker": document.metadata["ticker"],
            "fiscal_year": document.metadata["fiscal_year"],
            "section": document.metadata["section"],
            "score": document.metadata["vector_similarity"],
            "content": document.page_content,
        }
        for number, document in enumerate(passing, start=1)
    ]

    # Number the chunks for the prompt: "[1] NVDA FY2026 10-K, Risk Factors:"
    excerpt_blocks = []
    for source in sources:
        year = f"FY{source['fiscal_year']} " if source["fiscal_year"] else ""
        section_name = SECTION_NAMES.get(source["section"], source["section"])
        label = f"[{source['number']}] {source['ticker']} {year}10-K, {section_name}:"
        excerpt_blocks.append(f"{label}\n{source['content']}")

    return retrieved, passing, sources, "\n\n".join(excerpt_blocks)


# The one flow of the chat feature. The session check and the model are done HERE, before any
# response exists, so a missing session or key is a normal JSON error (404, 503). The returned
# generator then does the rest and yields NDJSON events as dicts:
#   sources, token..., done  (or error, when something fails after the stream has started)
def ask(
    db: Session,
    org_id: int,
    user_id: int,
    session_id: int,
    question: str,
    ticker: str | None,
    mode: str = "rag",
    trace_tags: list[str] | None = None,
) -> Iterator[dict]:
    # 1. Load the session with org_id AND user_id: a missing session, a colleague's session and
    # another organization's session all give the same 404
    chat_session = chat_repository.get_session(db, org_id, user_id, session_id)
    if chat_session is None:
        raise NotFoundError("Chat session not found")

    try:
        model = llm.get_chat_model()
    except ValueError:
        raise ServiceUnavailableError("Chat is disabled: OPENAI_API_KEY not set") from None

    def events() -> Iterator[dict]:
        try:
            # 2. The last messages of this session (the session was loaded through the scoped
            # query above, so reading its messages by id is safe)
            history = chat_repository.list_recent_messages(
                db, chat_session.id, settings.CHAT_HISTORY_MESSAGES
            )

            # 3. Rewrite a follow-up into a standalone question. No history, no LLM call
            standalone_question = question
            rewritten_question = None
            if history:
                history_text = "\n".join(
                    f"{'User' if message.role == 'user' else 'Assistant'}: {message.content}"
                    for message in history
                )
                rewritten = (REWRITE_PROMPT | model).invoke(
                    {"history": history_text, "question": question},
                    config=llm.get_trace_config(
                        "rag_chat",
                        user_id,
                        session_id,
                        ["rag", f"mode:{mode}", "step:rewrite", *(trace_tags or [])],
                    )
                    or None,
                )
                rewritten_question = rewritten.text.strip() or None
                if rewritten_question:
                    standalone_question = rewritten_question

            # 4. Retrieve inside the RAG scope, apply the relevance threshold and number the
            # sources (shared with the evaluation script)
            retrieved, passing, sources, excerpts = prepare_context(db, standalone_question, ticker)
            logger.info(
                "chat: %d chunks retrieved, %d passed the threshold (rewrite=%s)",
                len(retrieved),
                len(passing),
                rewritten_question is not None,
            )

            # The numbered sources go to the client first (empty list for "I don't know")
            yield {"type": "sources", "sources": sources}

            if not passing:
                # 5. Nothing relevant: fixed answer, no answer LLM call, no citations
                logger.info("chat: no relevant chunk, the answer model is NOT called")
                answer_text = NO_ANSWER
                answer_model = None
                yield {"type": "token", "text": NO_ANSWER}
            else:
                # 6. Stream the answer, forwarding every non-empty piece (OpenAI also sends empty
                # pieces at the start and the end)
                logger.info("chat: calling the answer model with %d excerpts", len(sources))
                answer_text = ""
                answer_model = settings.CHAT_MODEL
                for piece in (QA_PROMPT | model).stream(
                    {"excerpts": excerpts, "question": standalone_question},
                    config=llm.get_trace_config(
                        "rag_chat",
                        user_id,
                        session_id,
                        ["rag", f"mode:{mode}", "step:answer", *(trace_tags or [])],
                    )
                    or None,
                ):
                    if piece.text:
                        answer_text += piece.text
                        yield {"type": "token", "text": piece.text}
                if not answer_text.strip():
                    raise ValueError("The model returned an empty answer")

            # 7. Parse the markers [n] and [1, 2]. Numbers outside 1..n are ignored
            cited_numbers = set()
            for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer_text):
                for number_text in group.split(","):
                    number = int(number_text)
                    if 1 <= number <= len(passing):
                        cited_numbers.add(number)

            # Save everything in ONE transaction: nothing is saved when an earlier step failed
            chat_repository.create_message(
                db, chat_session.id, "user", question, rewritten_question, ticker
            )
            assistant_message = chat_repository.create_message(
                db, chat_session.id, "assistant", answer_text, model=answer_model
            )
            citations = []
            for number in sorted(cited_numbers):
                source = sources[number - 1]
                citations.append(
                    Citation(
                        message_id=assistant_message.id,
                        number=number,
                        chunk_id=source["chunk_id"],
                        score=source["score"],
                        filing_id=passing[number - 1].metadata["filing_id"],
                        ticker=source["ticker"],
                        fiscal_year=source["fiscal_year"],
                        section=source["section"],
                        content=source["content"],
                    )
                )
            chat_repository.create_citations(db, citations)
            if chat_session.title is None:
                chat_session.title = question.strip()[:100]
            chat_repository.touch_session(db, chat_session)
            db.commit()

            yield {
                "type": "done",
                "message_id": assistant_message.id,
                "cited_numbers": sorted(cited_numbers),
            }
        except (openai.OpenAIError, SQLAlchemyError, ValueError) as exc:
            # Only the class name is logged: an OpenAI error message can contain part of the key.
            # The client gets a generic text. Nothing was committed for this question
            db.rollback()
            logger.error("chat failed after the stream started: %s", type(exc).__name__)
            yield {"type": "error", "detail": "The answer could not be completed. Try again."}

    return events()


def create_session(db: Session, org_id: int, user_id: int) -> ChatSession:
    chat_session = chat_repository.create_session(db, org_id, user_id)
    audit_repository.create(
        db, org_id, user_id, action="chat_session.create", entity_id=chat_session.id
    )
    db.commit()
    return chat_session


def list_sessions(db: Session, org_id: int, user_id: int) -> list[ChatSession]:
    return chat_repository.list_sessions(db, org_id, user_id)


def get_session_detail(
    db: Session, org_id: int, user_id: int, session_id: int
) -> SessionDetailResponse:
    chat_session = chat_repository.get_session(db, org_id, user_id, session_id)
    if chat_session is None:
        raise NotFoundError("Chat session not found")

    rows = chat_repository.list_messages_with_citations(db, chat_session.id)
    # Answers written by an agent run get the run id (the messages come from the loaded session)
    run_ids = agent_repository.list_run_ids_by_answer_messages(
        db, [message.id for message, _ in rows]
    )
    messages = [
        MessageResponse(
            id=message.id,
            role=message.role,
            content=message.content,
            created_at=message.created_at,
            citations=[CitationResponse.model_validate(citation) for citation in citations],
            run_id=run_ids.get(message.id),
        )
        for message, citations in rows
    ]
    return SessionDetailResponse(
        id=chat_session.id,
        title=chat_session.title,
        created_at=chat_session.created_at,
        updated_at=chat_session.updated_at,
        messages=messages,
    )


def delete_session(db: Session, org_id: int, user_id: int, session_id: int) -> None:
    chat_session = chat_repository.get_session(db, org_id, user_id, session_id)
    if chat_session is None:
        raise NotFoundError("Chat session not found")

    chat_repository.delete_session(db, chat_session)
    audit_repository.create(db, org_id, user_id, action="chat_session.delete", entity_id=session_id)
    db.commit()
