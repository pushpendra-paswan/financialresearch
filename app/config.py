from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Values come from environment variables, or from the .env file if present.
    # The .env file also holds POSTGRES_* variables used only by docker compose,
    # so extras are ignored.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Financial Research Copilot"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # JWT settings. The secret has no default on purpose: generate one with `openssl rand -hex 32`
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Sent as the User-Agent on every SEC request. No default on purpose: the SEC requires a real
    # contact in the form "FinCopilot your.name@example.com"
    SEC_USER_AGENT: str
    # Raw downloaded files are saved here before parsing (relative to the project root)
    RAW_DATA_DIR: str = "data/raw"
    # Filing ingestion stores only filings filed within this many years
    FILINGS_LOOKBACK_YEARS: int = 3
    # Financial fact ingestion stores only periods that ended within this many years
    FINANCIALS_LOOKBACK_YEARS: int = 6
    # Which PriceProvider implementation to use (see clients/prices.py). Only "yfinance" for now
    PRICE_PROVIDER: str = "yfinance"
    # Every price run fetches this many years of daily bars (365 days per year)
    PRICES_LOOKBACK_YEARS: int = 5

    # Phase 2 and 3 (RAG and the agent) work only on these tickers (comma-separated) and only on
    # filings filed within this many years (365 days per year). This keeps embedding and LLM
    # costs small; Phase 1 ingestion still covers every company
    RAG_TICKERS: str = "AAPL,NVDA"
    RAG_LOOKBACK_YEARS: int = 2

    # Embeddings (2.2). The key is optional so the app and the tests start without it: an empty
    # key means "embeddings disabled". The vector size (1536) is a constant in
    # app/models/chunks.py, not a setting, because changing it needs a migration
    OPENAI_API_KEY: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    # Texts per embedding API request (this is the batch size, not the text chunk size)
    EMBEDDING_BATCH_SIZE: int = 100
    # Chunk size and overlap, in characters, for splitting each filing section
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200

    # Retrieval (2.3): each search (vector and full-text) returns this many candidates, the fused
    # list keeps RETRIEVAL_TOP_K chunks, and RRF_K is the standard reciprocal rank fusion constant
    RETRIEVAL_CANDIDATES_K: int = 20
    RETRIEVAL_TOP_K: int = 5
    RRF_K: int = 60

    # Chat (2.4). The chat model is OpenAI through LangChain's ChatOpenAI; the key is OPENAI_API_KEY
    # above (empty means chat is disabled). A chunk is used only when its vector_similarity is at
    # least RELEVANCE_THRESHOLD (real data: 0.095 for an off-topic question, 0.47 to 0.75 for real
    # ones). CHAT_HISTORY_MESSAGES is how many recent messages the question rewrite sees
    CHAT_MODEL: str = "gpt-5.4-mini"
    RELEVANCE_THRESHOLD: float = 0.30
    CHAT_HISTORY_MESSAGES: int = 6
    LLM_TIMEOUT_SECONDS: int = 60

    # Agent (3.2): the maximum number of model calls and the maximum wall time of one run. The
    # history given to the agent reuses CHAT_HISTORY_MESSAGES
    AGENT_MAX_STEPS: int = 8
    AGENT_TIMEOUT_SECONDS: int = 120
    # A write action waits this long for the user's decision, then it can no longer be approved
    APPROVAL_TTL_MINUTES: int = 60

    # Reranking (2.5): Cohere through LangChain's CohereRerank. An empty COHERE_API_KEY means
    # reranking is off (never logged). After fusion the best RERANK_CANDIDATES_K chunks are sent to
    # Cohere, which returns the top k. The timeout is short on purpose: a slow Cohere must fall
    # back to the fused order quickly (the Cohere SDK default is 300 seconds)
    COHERE_API_KEY: str = ""
    RERANK_ENABLED: bool = True
    RERANK_MODEL: str = "rerank-v4.0-fast"
    RERANK_CANDIDATES_K: int = 20
    RERANK_TIMEOUT_SECONDS: int = 10

    # Rate limits: requests per 60-second window. Login/register are counted per client IP,
    # every other protected route per authenticated user
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10
    RATE_LIMIT_API_PER_MINUTE: int = 120
    # Cached shared reads (companies, prices) expire after this many seconds
    CACHE_TTL_SECONDS: int = 600

    # No defaults on purpose: connection URLs must come from the environment
    DATABASE_URL: str
    REDIS_URL: str


settings = Settings()
