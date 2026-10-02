from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Values come from environment variables, or from the .env file if present.
    # The .env file also holds POSTGRES_* variables used only by docker compose,
    # so extras are ignored.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Financial Research Copilot"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"

    # No defaults on purpose: connection URLs must come from the environment
    DATABASE_URL: str
    REDIS_URL: str


settings = Settings()
