from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from app.config import settings


# The one place that creates LangChain model objects. The chat model is added here in 2.4.
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
