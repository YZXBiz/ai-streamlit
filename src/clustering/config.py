"""Define base settings for the application."""
# Author: Jackson Yang

# %% IMPORTS

from pydantic_settings import BaseSettings, SettingsConfigDict
from poethepoet.helpers.git import GitRepo
from pathlib import Path
from azure.identity import ClientSecretCredential


# %% Scretes SETTINGS
class ScretSettings(BaseSettings, extra="allow"):
    """Settings for the .env file."""

    model_config: dict = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )  # type: ignore

    AZURE_TENANT_ID: str
    AZURE_CLIENT_ID: str
    AZURE_CLIENT_SECRET: str

    # GitRepo can find the root dir where the .git file is located
    ROOT_DIR: str = str(GitRepo(seed_path=Path(".")).main_path)

    @property
    def AZURE_CREDS(self) -> ClientSecretCredential:
        """Return the azure credentials."""
        return ClientSecretCredential(
            tenant_id=self.AZURE_TENANT_ID,
            client_id=self.AZURE_CLIENT_ID,
            client_secret=self.AZURE_CLIENT_SECRET,
        )

    ACCOUNT_URL: str = "https://sartluse2peprod.blob.core.windows.net"
    CONTAINER_NAME: str = "slfsvc/twa07/experimentation"


SETTINGS = ScretSettings()
