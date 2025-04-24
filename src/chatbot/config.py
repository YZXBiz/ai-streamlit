"""
Application configuration for chatbot application.

This module provides a centralized configuration system using Pydantic Settings.
All environment variables and application settings are managed here.
"""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Application configuration container.

    Handles all settings, including database paths, API keys and feature flags.
    Automatically loads values from environment variables and .env files.
    """

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # API Keys and Secrets
    OPENAI_API_KEY: SecretStr = Field(
        description="OpenAI API key for embedding and completion endpoints",
    )

    # Database Settings
    DB_PATH: str = Field(
        default=":memory:",
        description="Path to DuckDB database file, :memory: for in-memory database",
    )

    # Chat Memory Settings
    MEMORY_TYPE: Literal["simple", "summary"] = Field(
        default="summary",
        description="Type of chat memory buffer to use (simple or summary)",
    )
    MEMORY_TOKEN_LIMIT: int = Field(
        default=4000,
        description="Maximum number of tokens to store in chat memory",
        ge=1000,
        le=16000,
    )

    # Logging - using str instead of Literal to allow case-insensitive validation
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level for application logs",
    )

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def check_api_key_not_empty(cls, v: SecretStr) -> SecretStr:
        """Validate that the OpenAI API key is not empty."""
        if not v.get_secret_value():
            raise ValueError("OpenAI API key cannot be empty")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Normalize and validate LOG_LEVEL."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {', '.join(valid_levels)}")
        return upper_v


# Create singleton instance to be imported by other modules
config = Config()
