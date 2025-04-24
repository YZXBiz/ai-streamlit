"""Application settings for the flat chatbot."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings container."""

    model_config = {"env_file": ".env", "extra": "ignore"}

    OPENAI_API_KEY: str
    DB_PATH: str = ":memory:"
    LOG_LEVEL: str = "INFO"


# Create singleton instance
SETTINGS = Settings()
