"""Pydantic models for assortment_chatbot configuration settings."""

import os
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureKeyVaultSettings(BaseSettings):
    """Azure Key Vault configuration settings."""

    # Key Vault connection
    KEY_VAULT_URL: str = Field(default="", description="Azure Key Vault URL")
    KEY_VAULT_ENABLED: bool = Field(
        default=False, description="Whether to use Azure Key Vault for secrets"
    )

    # Authentication method
    USE_MANAGED_IDENTITY: bool = Field(
        default=False, description="Whether to use managed identity authentication"
    )

    # Service principal authentication (only needed if not using managed identity)
    CLIENT_ID: str = Field(default="", description="Azure client ID for Key Vault access")
    CLIENT_SECRET: str = Field(default="", description="Azure client secret for Key Vault access")
    TENANT_ID: str = Field(default="", description="Azure tenant ID for Key Vault access")

    # Secrets to fetch (if empty, all secrets will be fetched)
    SECRET_NAMES: list[str] = Field(
        default=[], description="Specific secret names to fetch from Key Vault"
    )

    # Cache settings
    CACHE_SECRETS: bool = Field(default=True, description="Whether to cache secrets in memory")
    SECRET_REFRESH_INTERVAL: int = Field(
        default=3600, description="Seconds between secret refreshes if caching enabled"
    )

    @field_validator("KEY_VAULT_URL")
    def validate_vault_url(self, v: str, info) -> str:
        """Validate that the Key Vault URL has a valid format if enabled."""
        if info.data.get("KEY_VAULT_ENABLED", False) and not v:
            raise ValueError("KEY_VAULT_URL must be provided when KEY_VAULT_ENABLED is True")
        return v

    class Config:
        """Configuration for the settings model."""

        validate_assignment = True


class AppSettings(BaseSettings):
    """Global application settings with automatic .env loading."""

    # LLM Provider settings
    # OpenAI settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-3.5-turbo", description="OpenAI model name")

    # Azure OpenAI
    AZURE_OPENAI_KEY: str = Field(default="", description="Azure OpenAI API key")
    AZURE_OPENAI_ENDPOINT: str = Field(default="", description="Azure OpenAI endpoint")
    AZURE_DEPLOYMENT_NAME: str = Field(
        default="", description="Azure deployment name for chat completions"
    )

    # Anthropic settings
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
    ANTHROPIC_MODEL: str = Field(default="claude-3-sonnet", description="Anthropic model name")

    # Active provider
    ACTIVE_PROVIDER: Literal["openai", "azure", "anthropic"] = Field(
        default="openai", description="Active LLM provider to use"
    )

    # Database settings
    DUCKDB_PATH: str = Field(default=":memory:", description="DuckDB database path")
    MAX_ROWS_PREVIEW: int = Field(default=5, ge=1, le=100)

    # Agent settings
    SQL_AGENT_MODEL: str = Field(
        default="gpt-3.5-turbo", description="Model to use for SQL generation agent"
    )
    INTERPRETER_MODEL: str = Field(
        default="gpt-3.5-turbo", description="Model to use for result interpretation agent"
    )
    TEMPERATURE: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Temperature setting for LLM responses"
    )
    STREAMING: bool = Field(default=True, description="Whether to stream LLM responses")
    QUERY_TIMEOUT: int = Field(
        default=int(os.getenv("QUERY_TIMEOUT", "20")),
        ge=1,
        le=300,
        description="Timeout in seconds for query processing",
    )

    # App settings
    DEBUG_MODE: bool = Field(
        default=os.getenv("DEBUG", "false").lower() == "true", description="Enable debug mode"
    )
    ENVIRONMENT: Literal["development", "production", "test"] = Field(
        default=os.getenv("ENVIRONMENT", "development"), description="Application environment"
    )

    # Azure Key Vault settings
    key_vault: AzureKeyVaultSettings = Field(default_factory=AzureKeyVaultSettings)

    # Use Pydantic's .env file loading
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )

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
        return {"db_path": self.DUCKDB_PATH, "max_rows_preview": self.MAX_ROWS_PREVIEW}

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
            "query_timeout": self.QUERY_TIMEOUT,
            "active_provider": self.ACTIVE_PROVIDER,
        }

    @property
    def azure_vault_settings(self) -> dict:
        """Get Azure Key Vault settings as a dictionary.

        Returns:
            Dictionary with Azure Key Vault settings
        """
        return {
            "vault_url": self.key_vault.KEY_VAULT_URL,
            "enabled": self.key_vault.KEY_VAULT_ENABLED,
            "use_managed_identity": self.key_vault.USE_MANAGED_IDENTITY,
            "client_id": self.key_vault.CLIENT_ID,
            "client_secret": self.key_vault.CLIENT_SECRET,
            "tenant_id": self.key_vault.TENANT_ID,
            "secret_names": self.key_vault.SECRET_NAMES,
            "cache_secrets": self.key_vault.CACHE_SECRETS,
            "refresh_interval": self.key_vault.SECRET_REFRESH_INTERVAL,
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
