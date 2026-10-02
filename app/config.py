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

    # No defaults on purpose: connection URLs must come from the environment
    DATABASE_URL: str
    REDIS_URL: str


settings = Settings()
