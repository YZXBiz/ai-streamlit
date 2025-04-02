"""Configuration management for the clustering pipeline."""

from pathlib import Path
from typing import Any, Dict, Optional

from azure.identity import ClientSecretCredential
from poethepoet.helpers.git import GitRepo
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecretSettings(BaseSettings):
    """Settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # Azure credentials
    AZURE_TENANT_ID: str
    AZURE_CLIENT_ID: str
    AZURE_CLIENT_SECRET: str

    # Storage settings
    ACCOUNT_URL: str = "https://sartluse2peprod.blob.core.windows.net"
    CONTAINER_NAME: str = "slfsvc/twa07/experimentation"

    # GitRepo can find the root dir where the .git file is located
    ROOT_DIR: str = str(GitRepo(seed_path=Path(".")).main_path)

    @property
    def AZURE_CREDS(self) -> ClientSecretCredential:
        """Return the azure credentials.

        Returns:
            ClientSecretCredential: Azure credentials
        """
        return ClientSecretCredential(
            tenant_id=self.AZURE_TENANT_ID,
            client_id=self.AZURE_CLIENT_ID,
            client_secret=self.AZURE_CLIENT_SECRET,
        )


class JobSettings(BaseSettings, strict=True, frozen=True, extra="forbid"):
    """Base class for all job settings.

    All job types should inherit from this class and add their specific settings.
    """

    KIND: str


class AppConfig:
    """Application configuration manager.

    This class loads and manages configuration from:
    - Environment variables
    - .env file
    - Configuration files
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to configuration files (optional)
        """
        self.secrets = SecretSettings()
        self.config_path = config_path or "configs"
        self._job_settings: Dict[str, Any] = {}

    def load_job_config(self, job_name: str) -> Dict[str, Any]:
        """Load job configuration from YAML file.

        Args:
            job_name: Name of the job

        Returns:
            Dict containing job configuration
        """
        import yaml

        config_file = Path(self.config_path) / f"{job_name}.yml"

        if not config_file.exists():
            # Try with default extension
            config_file = Path(self.config_path) / f"{job_name}.yaml"

        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found for job {job_name}")

        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        self._job_settings[job_name] = config
        return config

    def get_job_config(self, job_name: str) -> Dict[str, Any]:
        """Get job configuration.

        Args:
            job_name: Name of the job

        Returns:
            Dict containing job configuration
        """
        if job_name not in self._job_settings:
            return self.load_job_config(job_name)
        return self._job_settings[job_name]

    def get_env(self) -> str:
        """Get current environment.

        Returns:
            String representing current environment (dev, staging, prod)
        """
        import os

        return os.environ.get("DAGSTER_ENV", "dev")


# Global instance
CONFIG = AppConfig()
