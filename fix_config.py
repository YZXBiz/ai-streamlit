#!/usr/bin/env python3
"""Script to regenerate the config.py file with fixed settings."""

import os

# Ensure we're using a hardcoded value for ACCESS_TOKEN_EXPIRE_MINUTES
config_content = '''"""Application configuration from environment variables."""

import os
import tempfile

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings class."""

    # API configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PandasAI Chat API"
    OPENAI_API_KEY: str = Field(..., description="OpenAI API key")

    # Database configuration
    DB_HOST: str = Field("localhost", description="Database hostname")
    DB_PORT: int = Field(5432, description="Database port")
    DB_USER: str = Field("postgres", description="Database username")
    DB_PASSWORD: str = Field("postgres", description="Database password")
    DB_NAME: str = Field("chatbot", description="Database name")
    DB_ECHO: bool = Field(False, description="Echo SQL statements to console")
    DB_POOL_SIZE: int = Field(5, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(10, description="Maximum database connection overflow")
    SQL_ECHO: bool = Field(False, description="Echo SQL statements to console")

    # Database URLs
    DATABASE_URL: str = Field(
        "postgresql://postgres:postgres@localhost:5432/chatbot",
        description="Database URL for synchronous connections",
    )
    ASYNC_DATABASE_URL: str = Field(
        "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot",
        description="Database URL for asynchronous connections",
    )

    # DuckDB configuration (for local analysis)
    DUCKDB_PATH: str = Field(":memory:", description="DuckDB database path or :memory:")
    SQL_QUERY_TIMEOUT: int = Field(30, description="SQL query timeout in seconds")

    # CORS configuration
    CORS_ORIGINS: list[str] = Field(
        ["http://localhost:8501", "http://frontend:8501"],
        description="List of allowed CORS origins",
    )

    # Authentication
    SECRET_KEY: str = Field(
        "supersecretkey",
        description="Secret key for JWT tokens (change this in production!)",
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # Token expiry time in minutes

    # PandasAI configuration
    MEMORY_SIZE: int = Field(10, description="Number of conversation turns to keep in memory")
    ENFORCE_PRIVACY: bool = Field(True, description="Whether to enforce privacy mode in PandasAI")
    ENABLE_CACHE: bool = Field(True, description="Whether to enable caching in PandasAI")
    MAX_RETRIES: int = Field(3, description="Maximum number of retries for failed queries")
    CHARTS_DIR: str = Field(
        os.path.join(tempfile.gettempdir(), "pandasai_charts"),
        description="Directory for chart storage",
    )

    # Storage configuration
    DATA_DIR: str = Field("./data", description="Data directory for file storage")
    LOGS_DIR: str = Field("./logs", description="Directory for log files")
    STORAGE_PATH: str = Field("./data/uploads", description="Path for file uploads")

    # Vector store configuration
    VECTOR_STORE_DIR: str = Field(
        "./data/vector_store", description="Directory for vector store indices"
    )
    EMBEDDING_MODEL: str = Field("all-MiniLM-L6-v2", description="Model to use for text embeddings")

    # Logging configuration
    LOG_LEVEL: str = Field("INFO", description="Logging level")

    # Allow env file and case-sensitive settings
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")


# Create global settings instance
settings = Settings()
'''

# Write the content to the config file
with open(os.path.join("app", "core", "config.py"), "w") as f:
    f.write(config_content)

print("Config file has been reset.")
