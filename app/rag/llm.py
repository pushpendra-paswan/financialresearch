import functools
import logging

import cohere
from langchain_cohere import CohereRerank
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tracers.langchain import LangChainTracer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import Client
from urllib3.util import Retry

from app.config import settings

logger = logging.getLogger(__name__)


# The one place that creates LangChain model objects.
def get_embeddings() -> Embeddings:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    # chunk_size is the number of texts per API request (the batch size), not the text chunk size.
    # check_embedding_ctx_length=False sends the plain text instead of tiktoken token arrays:
    # our chunks are at most about 600 tokens (the limit is 8191), and it means the worker never
    # has to download a tiktoken file. dimensions is not passed, so a wrong EMBEDDING_MODEL
    # produces a vector of another size, which embed_filings reports clearly
    return OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        chunk_size=settings.EMBEDDING_BATCH_SIZE,
        check_embedding_ctx_length=False,
        # Without it the OpenAI SDK waits 10 minutes: the agent's search_filings embeds queries
        # inside a run that has its own time limit
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


def get_chat_model() -> BaseChatModel:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    # No temperature: gpt-5.x models accept only the default (LangChain drops any other value).
    # reasoning_effort is left unset: the default call used no reasoning tokens. A model call
    # that hangs ends after LLM_TIMEOUT_SECONDS; max_retries covers transient OpenAI errors
    return ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.OPENAI_API_KEY,
        timeout=settings.LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )


def get_reranker(top_n: int) -> CohereRerank:
    if not settings.COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY is not set")

    # We build the Cohere client ourselves because CohereRerank has no timeout option and the
    # SDK default is 300 seconds. Creating it makes no network call. Documents are never
    # truncated: a chunk is at most about 400 tokens, far below Cohere's 4000-token default limit
    client = cohere.ClientV2(settings.COHERE_API_KEY, timeout=settings.RERANK_TIMEOUT_SECONDS)
    return CohereRerank(model=settings.RERANK_MODEL, client=client, top_n=top_n)


# Called by the LangSmith client from its background thread when an upload fails (a bad key, a
# rate limit, an outage). Only the class name is logged: the library's own messages can contain a
# part of the key
def log_tracing_error(error: Exception) -> None:
    logger.warning("LangSmith tracing failed: %s", type(error).__name__)


# One client per process: every Client starts its own background thread. Creating it makes no
# network call. The short timeouts and no retries (a trace is best effort) keep a LangSmith outage
# from holding the process open for minutes at exit (the default drain took about 85 seconds with
# an unreachable endpoint, this one about 7 seconds after the bounded flush). The library logs
# failures with a partly masked key, so its logger is silenced and log_tracing_error reports
# instead
@functools.cache
def get_langsmith_client() -> Client:
    logging.getLogger("langsmith").setLevel(logging.CRITICAL)
    return Client(
        api_key=settings.LANGSMITH_API_KEY,
        api_url=settings.LANGSMITH_ENDPOINT or None,
        timeout_ms=(2000, 10000),
        retry_config=Retry(total=0),
        tracing_error_callback=log_tracing_error,
    )


# The ONLY way a call is traced: the caller merges the returned fragment into the config it gives
# to .invoke(), .stream() or graph.stream(). It is {} when tracing is off (empty
# LANGSMITH_API_KEY), so an untraced call is exactly what it was. Global tracing
# (LANGSMITH_TRACING) is never set, so a call without this fragment is never traced. The metadata
# holds numeric ids only: never an email, a key or the question. "thread_id" makes LangSmith group
# the traces of one chat session in its Threads view (our own configurable thread_id, the agent
# run id, is overridden because an explicit metadata key wins)
def get_trace_config(
    name: str, user_id: int, session_id: int, tags: list[str], metadata: dict | None = None
) -> dict:
    if not settings.LANGSMITH_API_KEY:
        return {}

    # Tracing must never break a request: whatever building the tracer raises, the call goes on
    # untraced. Exception (not a narrower class) is deliberate: the client and the tracer come
    # from a third-party package whose errors we do not control
    try:
        tracer = LangChainTracer(
            client=get_langsmith_client(), project_name=settings.LANGSMITH_PROJECT
        )
    except Exception as exc:
        logger.warning("tracing is off for this call: %s", type(exc).__name__)
        return {}

    return {
        "callbacks": [tracer],
        "run_name": name,
        "tags": list(tags),
        "metadata": {
            "user_id": str(user_id),
            "thread_id": f"chat-{session_id}",
            **(metadata or {}),
        },
    }


# The tracer uploads in a background thread, so a short script must wait for it before it exits.
# The wait is bounded: an unreachable LangSmith cannot hold the script for long
def flush_traces() -> None:
    if not settings.LANGSMITH_API_KEY:
        return
    try:
        get_langsmith_client().flush(timeout=10)
    except Exception as exc:
        logger.warning("flushing traces failed: %s", type(exc).__name__)
