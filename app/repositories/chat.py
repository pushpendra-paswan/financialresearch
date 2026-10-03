# Chat sessions are private and personal: the session reads take org_id AND user_id.
# Message and citation reads take a session_id that the service has ALREADY loaded through
# get_session (org_id and user_id), the same pattern as the watchlist item queries in 1.3.
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, ChatSession, Citation


def create_session(db: Session, org_id: int, user_id: int) -> ChatSession:
    chat_session = ChatSession(org_id=org_id, user_id=user_id)
    db.add(chat_session)
    db.flush()
    return chat_session


def get_session(db: Session, org_id: int, user_id: int, session_id: int) -> ChatSession | None:
    statement = select(ChatSession).where(
        ChatSession.id == session_id, ChatSession.org_id == org_id, ChatSession.user_id == user_id
    )
    return db.execute(statement).scalar_one_or_none()


def list_sessions(db: Session, org_id: int, user_id: int) -> list[ChatSession]:
    # Most recently used first, at most 50
    statement = (
        select(ChatSession)
        .where(ChatSession.org_id == org_id, ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
        .limit(50)
    )
    return list(db.execute(statement).scalars().all())


def delete_session(db: Session, chat_session: ChatSession) -> None:
    # The caller loaded the session with get_session, so ownership is checked. The database
    # cascade removes its messages and their citations
    db.delete(chat_session)
    db.flush()


def touch_session(db: Session, chat_session: ChatSession) -> None:
    # A Python timestamp, not now(): now() is the start of the transaction, which is the start of
    # the stream, and the sessions list is ordered by this value
    chat_session.updated_at = datetime.now(UTC)
    db.flush()


def list_recent_messages(db: Session, session_id: int, limit: int) -> list[ChatMessage]:
    # The last `limit` messages, returned oldest first
    statement = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
    )
    return list(reversed(db.execute(statement).scalars().all()))


def list_messages_with_citations(
    db: Session, session_id: int
) -> list[tuple[ChatMessage, list[Citation]]]:
    messages = db.execute(
        select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)
    ).scalars()
    messages = list(messages)

    citations_by_message: dict[int, list[Citation]] = {message.id: [] for message in messages}
    if messages:
        citations = db.execute(
            select(Citation)
            .where(Citation.message_id.in_(citations_by_message))
            .order_by(Citation.message_id, Citation.number)
        ).scalars()
        for citation in citations:
            citations_by_message[citation.message_id].append(citation)

    return [(message, citations_by_message[message.id]) for message in messages]


def create_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
    rewritten_question: str | None = None,
    ticker: str | None = None,
    model: str | None = None,
) -> ChatMessage:
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        rewritten_question=rewritten_question,
        ticker=ticker,
        model=model,
    )
    db.add(message)
    db.flush()
    return message


def create_citations(db: Session, citations: list[Citation]) -> None:
    db.add_all(citations)
    db.flush()
