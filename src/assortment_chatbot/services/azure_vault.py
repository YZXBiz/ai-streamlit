"""Azure Key Vault integration for secure secrets management.

This module provides functions to authenticate and retrieve secrets from Azure Key Vault
using various authentication methods including client credentials and managed identities.
"""

from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.identity import ClientSecretCredential, DefaultAzureCredential, ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient

from assortment_chatbot.utils.logging import get_logger

logger = get_logger(__name__)


class AzureKeyVaultClient:
    """Client for securely accessing secrets from Azure Key Vault."""

    def __init__(
        self,
        vault_url: str,
        client_id: str | None = None,
        client_secret: str | None = None,
        tenant_id: str | None = None,
        use_managed_identity: bool = False,
    ):
        """Initialize Azure Key Vault client.

        Args:
            vault_url: The URL of the Azure Key Vault
            client_id: Azure client ID for service principal auth
            client_secret: Azure client secret for service principal auth
            tenant_id: Azure tenant ID for service principal auth
            use_managed_identity: Whether to use managed identity authentication
        """
        self.vault_url = vault_url

        # Determine which credential to use
        if use_managed_identity:
            self.credential = ManagedIdentityCredential()
            logger.info("Using Managed Identity for Azure Key Vault authentication")
        elif client_id and client_secret and tenant_id:
            self.credential = ClientSecretCredential(
                tenant_id=tenant_id, client_id=client_id, client_secret=client_secret
            )
            logger.info("Using Service Principal for Azure Key Vault authentication")
        else:
            # DefaultAzureCredential tries multiple authentication methods
            # including environment variables, managed identity, and Visual Studio Code
            self.credential = DefaultAzureCredential()
            logger.info("Using DefaultAzureCredential for Azure Key Vault authentication")

        try:
            # Create the client
            self.client = SecretClient(vault_url=self.vault_url, credential=self.credential)
            logger.info(f"Connected to Azure Key Vault: {vault_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Azure Key Vault: {str(e)}")
            raise

    def get_secret(self, secret_name: str, version: str | None = None) -> str | None:
        """Get a secret from Azure Key Vault.

        Args:
            secret_name: Name of the secret to retrieve
            version: Optional specific version of the secret

        Returns:
            The secret value or None if not found
        """
        try:
            secret = self.client.get_secret(name=secret_name, version=version)
            return secret.value
        except ResourceNotFoundError:
            logger.warning(f"Secret '{secret_name}' not found in Key Vault")
            return None
        except HttpResponseError as e:
            logger.error(f"Error retrieving secret '{secret_name}': {str(e)}")
            return None

    def list_secrets(self) -> list[str]:
        """List all secret names in the vault.

        Returns:
            List of secret names
        """
        try:
            return [secret.name for secret in self.client.list_properties_of_secrets()]
        except Exception as e:
            logger.error(f"Error listing secrets: {str(e)}")
            return []

    def get_all_secrets(self) -> dict[str, str]:
        """Fetch all secrets from Azure Key Vault.

        Returns:
            Dictionary of secret names to values
        """
        secrets = {}

        try:
            # Get names of all secrets in the vault
            secret_names = self.list_secrets()

            # Fetch each secret
            for name in secret_names:
                value = self.get_secret(name)
                if value is not None:
                    secrets[name] = value
        except Exception as e:
            logger.error(f"Error retrieving all secrets: {str(e)}")

        return secrets


def get_vault_secrets(
    vault_url: str,
    client_id: str | None = None,
    client_secret: str | None = None,
    tenant_id: str | None = None,
    use_managed_identity: bool = False,
    secret_names: list[str] | None = None,
) -> dict[str, str]:
    """Helper function to retrieve secrets from Azure Key Vault.

    Args:
        vault_url: The URL of the Azure Key Vault
        client_id: Azure client ID for service principal authentication
        client_secret: Azure client secret for service principal authentication
        tenant_id: Azure tenant ID for service principal authentication
        use_managed_identity: Whether to use managed identity authentication
        secret_names: Optional list of specific secret names to retrieve

    Returns:
        Dictionary mapping secret names to their values
    """
    if not vault_url:
        return {}

    try:
        # Initialize the client
        client = AzureKeyVaultClient(
            vault_url=vault_url,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
            use_managed_identity=use_managed_identity,
        )

        # Retrieve specific secrets if names are provided
        if secret_names:
            secrets = {}
            for name in secret_names:
                value = client.get_secret(name)
                if value is not None:
                    secrets[name] = value
            return secrets

        # Otherwise, retrieve all secrets
        return client.get_all_secrets()

    except Exception as e:
        logger.error(f"Failed to retrieve secrets from Azure Key Vault: {str(e)}")
        return {}
