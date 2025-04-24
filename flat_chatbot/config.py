"""Application settings for the flat chatbot."""

from dataclasses import dataclass
from pathlib import Path

from pydantic_settings import BaseSettings
from llama_index.core.prompts import PromptTemplate

@dataclass
class Paths:
    """Application path configuration."""

    # Root directory - parent of the flat_chatbot package
    root: Path = Path(__file__).parent.parent

    # Data directories
    data: Path = root / "data"
    temp: Path = data / "temp"

    # Log directory
    logs: Path = root / "logs"

    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        self.data.mkdir(parents=True, exist_ok=True)
        self.temp.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings container."""

    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4.1-nano"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TEMPERATURE: float = 0.0
    DB_PATH: str = ":memory:"
    LOG_LEVEL: str = "INFO"
    TOKEN_LIMIT: int = 4000
    MEMORY_TYPE: str = "summary"
    QUERY_TIMEOUT: int = 20  # seconds
    
    # prompt templates (default prompts from llama_index are good enough)
    USER_TEXT_TO_SQL_PROMPT: PromptTemplate | None = None
    USER_RESPONSE_SYNTHESIS_PROMPT: PromptTemplate | None = None
    USER_REFINE_SYNTHESIS_PROMPT: PromptTemplate | None = None
    # NOTE: Please check the original prompts in the llama_index library and follow the same format
    # Default prompts are: 
    # DEFAULT_TEXT_TO_SQL_PROMPT
    # DEFAULT_RESPONSE_SYNTHESIS_PROMPT_V2
    # DEFAULT_REFINE_PROMPT

    PATHS: Paths = Paths()

    class Config:
        env_file = ".env"
        extra = "ignore"


# Create singleton instance
SETTINGS = Settings()
