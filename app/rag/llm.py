import cohere
from langchain_cohere import CohereRerank
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import settings


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
