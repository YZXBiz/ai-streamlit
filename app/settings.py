from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and type checking.
    """

    # API Keys
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")

    # Database Configuration
    db_path: str = Field(":memory:", description="Path to database")

    # Chat Memory Settings
    memory_type: str = Field("simple", description="Type of chat memory to use")
    memory_token_limit: int = Field(4000, description="Token limit for chat memory")

    # Logging
    log_level: str = Field("INFO", description="Logging level")
    logs_dir: str = Field("./logs", description="Directory for log files")

    # Authentication
    cookie_secret: str = Field(..., description="Secret key for cookie encryption")
    password_salt: str = Field("default_salt", description="Salt for password hashing")
    default_username: str = Field("admin", description="Default username for login")
    default_password: str = Field("password", description="Default password for login")

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


# Create a global instance of settings
settings = Settings()
