"""Settings manager for Streamlit integration with Pydantic settings.

This module provides utility functions to sync environment variables between
Streamlit's st.secrets, Azure Key Vault, and our pydantic AppSettings.
"""

import logging
import os
import threading
import time
from typing import Any

import streamlit as st

from assortment_chatbot.config.settings import AppSettings, get_settings
from assortment_chatbot.services.azure_vault import get_vault_secrets

# Set up logging
logger = logging.getLogger(__name__)

# Global variables for secret caching
_secret_cache: dict[str, str] = {}
_last_refresh_time: float = 0
_secret_lock = threading.Lock()


def _get_secrets_from_vault(settings: AppSettings) -> dict[str, str]:
    """Get secrets from Azure Key Vault.

    Args:
        settings: Application settings

    Returns:
        Dictionary of secrets from Key Vault
    """
    vault_settings = settings.azure_vault_settings

    if not vault_settings["enabled"] or not vault_settings["vault_url"]:
        return {}

    try:
        return get_vault_secrets(
            vault_url=vault_settings["vault_url"],
            client_id=vault_settings["client_id"],
            client_secret=vault_settings["client_secret"],
            tenant_id=vault_settings["tenant_id"],
            use_managed_identity=vault_settings["use_managed_identity"],
            secret_names=vault_settings["secret_names"] if vault_settings["secret_names"] else None,
        )
    except Exception as e:
        logger.error(f"Failed to retrieve secrets from Azure Key Vault: {str(e)}")
        return {}


def _get_cached_secrets(settings: AppSettings) -> dict[str, str]:
    """Get secrets from cache or Azure Key Vault if cache expired.

    Args:
        settings: Application settings

    Returns:
        Dictionary of secrets
    """
    global _secret_cache, _last_refresh_time

    vault_settings = settings.azure_vault_settings

    # If caching is disabled, always fetch fresh secrets
    if not vault_settings["cache_secrets"]:
        return _get_secrets_from_vault(settings)

    current_time = time.time()
    refresh_interval = vault_settings["refresh_interval"]

    with _secret_lock:
        # Check if cache needs refreshing
        if not _secret_cache or (current_time - _last_refresh_time) > refresh_interval:
            logger.info("Refreshing secrets from Azure Key Vault")
            _secret_cache = _get_secrets_from_vault(settings)
            _last_refresh_time = current_time

    return _secret_cache.copy()


def sync_settings_with_streamlit() -> AppSettings:
    """Synchronize settings between Streamlit secrets, Azure Key Vault, and Pydantic AppSettings.

    This function handles the complete flow of:
    1. Loading settings from environment variables
    2. Merging in Streamlit secrets
    3. Fetching secrets from Azure Key Vault
    4. Recreating settings with the updated environment

    Returns:
        AppSettings: The synchronized settings object
    """
    # Get initial settings
    settings = get_settings()

    # 1. Copy environment variables from Streamlit secrets to os.environ if they don't exist
    try:
        for key in st.secrets:
            # Special handling for nested secrets
            if isinstance(st.secrets[key], dict):
                for nested_key, nested_value in st.secrets[key].items():
                    prefixed_key = f"{key.upper()}_{nested_key.upper()}"
                    if prefixed_key not in os.environ and isinstance(nested_value, str):
                        os.environ[prefixed_key] = nested_value
            # Regular top-level secrets
            elif key not in os.environ and isinstance(st.secrets[key], str):
                os.environ[key] = st.secrets[key]
    except (AttributeError, RuntimeError):
        # Handle cases where st.secrets is not available (non-Streamlit environment)
        logger.info("Streamlit secrets not available")

    # 2. Recreate settings to pick up any new values from os.environ
    settings = AppSettings()

    # 3. If Azure Key Vault is enabled, fetch secrets and update environment variables
    if settings.key_vault.KEY_VAULT_ENABLED:
        vault_secrets = _get_cached_secrets(settings)

        # Update environment variables with vault secrets
        for key, value in vault_secrets.items():
            # Only update if the environment variable doesn't already exist
            # This gives priority to explicit environment variables
            if key not in os.environ:
                os.environ[key] = value

        # Recreate settings again to pick up values from Key Vault
        settings = AppSettings()

    # 4. Store all settings in session state for easy access in Streamlit
    try:
        if "settings" not in st.session_state:
            st.session_state.settings = settings
    except (AttributeError, RuntimeError):
        # Handle cases where st.session_state is not available (non-Streamlit environment)
        logger.info("Streamlit session state not available")

    return settings


def get_streamlit_setting(key: str, default: str | None = None) -> Any:
    """Get a setting from Streamlit secrets or environment variables.

    Attempts to retrieve a value from:
    1. Streamlit session state settings
    2. Streamlit secrets
    3. Environment variables
    4. Default value

    Args:
        key: The setting key to retrieve
        default: Optional default value if setting is not found

    Returns:
        The setting value or default
    """
    # Check session state first
    try:
        if "settings" in st.session_state:
            pydantic_settings = st.session_state.settings
            if hasattr(pydantic_settings, key):
                return getattr(pydantic_settings, key)
    except (AttributeError, RuntimeError):
        pass

    # Then check Streamlit secrets
    try:
        if key in st.secrets:
            return st.secrets[key]
    except (AttributeError, RuntimeError):
        pass

    # Then check environment variables
    if key in os.environ:
        return os.environ[key]

    # Finally return default
    return default


def display_debug_settings() -> None:
    """Display current settings in debug mode.

    This function shows all non-sensitive settings in the Streamlit UI
    when debug mode is enabled.
    """
    try:
        if not st.session_state.get("debug_mode", False):
            return

        settings = st.session_state.settings

        # Show debug info in sidebar
        with st.sidebar.expander("Debug: Environment Settings"):
            st.write(f"Environment: {settings.ENVIRONMENT}")
            st.write(f"Debug Mode: {settings.DEBUG_MODE}")
            st.write(f"DuckDB Path: {settings.DUCKDB_PATH}")
            st.write(f"Query Timeout: {settings.QUERY_TIMEOUT}s")

            # Show Key Vault status
            st.write(f"Azure Key Vault Enabled: {settings.key_vault.KEY_VAULT_ENABLED}")
            if settings.key_vault.KEY_VAULT_ENABLED:
                st.write(f"Key Vault URL: {settings.key_vault.KEY_VAULT_URL}")
                st.write(f"Using Managed Identity: {settings.key_vault.USE_MANAGED_IDENTITY}")

            # Show sensitive settings as masked values
            for setting_name in ["OPENAI_API_KEY", "AZURE_OPENAI_KEY", "ANTHROPIC_API_KEY"]:
                api_key = getattr(settings, setting_name, "")
                masked_key = "••••" + api_key[-4:] if api_key and len(api_key) > 4 else "Not set"
                st.write(f"{setting_name}: {masked_key}")
    except (AttributeError, RuntimeError):
        # Handle cases where Streamlit context is not available
        pass
