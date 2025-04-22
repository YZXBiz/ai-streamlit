"""Pydantic models for dashboard configuration settings."""

import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):
    """Global application settings with automatic .env loading."""
    
    # OpenAI settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4.1-nano", description="OpenAI model name")
    
    # Database settings
    DUCKDB_PATH: str = Field(default=":memory:", description="DuckDB database path")
    MAX_ROWS_PREVIEW: int = Field(default=5, ge=1, le=100)
    
    # Agent settings
    SQL_AGENT_MODEL: str = Field(default="openai:gpt-4.1-nano", description="Model to use for SQL generation agent")
    INTERPRETER_MODEL: str = Field(default="openai:gpt-4.1-nano", description="Model to use for result interpretation agent")
    TEMPERATURE: float = Field(default=0.1, ge=0.0, le=1.0, description="Temperature setting for LLM responses")
    STREAMING: bool = Field(default=True, description="Whether to stream LLM responses")
    QUERY_TIMEOUT: int = Field(default=int(os.getenv("QUERY_TIMEOUT", "20")), ge=1, le=300, description="Timeout in seconds for query processing")
    
    # App settings
    DEBUG_MODE: bool = Field(default=os.getenv("DEBUG", "false").lower() == "true", description="Enable debug mode")
    ENVIRONMENT: Literal["development", "production", "test"] = Field(
        default=os.getenv("ENVIRONMENT", "development"), 
        description="Application environment"
    )
    
    # Use Pydantic's .env file loading
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', case_sensitive=True)
    
    @property
    def openai_api_key(self) -> str:
        """Get the OpenAI API key.
        
        Returns:
            The API key as a string or empty string if not set
        """
        return self.OPENAI_API_KEY
    
    @property
    def duckdb_settings(self) -> dict:
        """Get DuckDB settings as a dictionary.
        
        Returns:
            Dictionary with DuckDB settings
        """
        return {
            "db_path": self.DUCKDB_PATH,
            "max_rows_preview": self.MAX_ROWS_PREVIEW
        }
    
    @property
    def agent_settings(self) -> dict:
        """Get agent settings as a dictionary.
        
        Returns:
            Dictionary with agent settings
        """
        return {
            "sql_agent_model": self.SQL_AGENT_MODEL,
            "interpreter_model": self.INTERPRETER_MODEL,
            "temperature": self.TEMPERATURE,
            "streaming": self.STREAMING,
            "query_timeout": self.QUERY_TIMEOUT
        }

# Create a global settings instance
def get_settings() -> AppSettings:
    """Load and return application settings.
    
    Settings are loaded from environment variables and .env file
    when available, otherwise default values are used.
    
    Returns:
        Application settings instance
    """
    return AppSettings()


if __name__ == "__main__":
    settings = get_settings()
    print(settings.model_dump_json(indent=2))