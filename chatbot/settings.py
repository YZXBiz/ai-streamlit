"""Application settings for the flat chatbot."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings, extra="ignore"):
    """Application settings container."""

    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str
    DB_PATH: str = ":memory:"
    LOG_LEVEL: str = "INFO"


# Create singleton instance
SETTINGS = Settings()
