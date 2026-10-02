from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Values come from environment variables, or from the .env file if present
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "Financial Research Copilot"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"


settings = Settings()
