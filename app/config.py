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
