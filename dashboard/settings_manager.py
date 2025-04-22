"""Settings manager for Streamlit integration with Pydantic settings.

This module provides utility functions to sync environment variables between
Streamlit's st.secrets and our pydantic AppSettings.
"""

import os
import streamlit as st
from typing import Optional

from dashboard.settings import AppSettings, get_settings

def sync_settings_with_streamlit() -> AppSettings:
    """Synchronize settings between Streamlit secrets and Pydantic AppSettings.
    
    This function handles bidirectional sync between:
    1. Environment variables
    2. Streamlit secrets (st.secrets)
    3. Pydantic AppSettings
    
    Returns:
        AppSettings: The synchronized settings object
    """
    # Get current settings
    settings = get_settings()
    
    # 1. Copy environment variables from secrets to os.environ if they don't exist
    for key in st.secrets:
        if key not in os.environ and isinstance(st.secrets[key], str):
            os.environ[key] = st.secrets[key]
    
    # 2. Recreate settings to pick up any new values from os.environ
    settings = AppSettings()
    
    # 3. Store all settings in session state for easy access
    if "settings" not in st.session_state:
        st.session_state.settings = settings
    
    return settings

def get_streamlit_setting(key: str, default: Optional[str] = None) -> str:
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
    if "settings" in st.session_state:
        pydantic_settings = st.session_state.settings
        if hasattr(pydantic_settings, key):
            return getattr(pydantic_settings, key)
    
    # Then check Streamlit secrets
    if key in st.secrets:
        return st.secrets[key]
    
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
    if not st.session_state.get("debug_mode", False):
        return
    
    settings = st.session_state.settings
    
    # Show debug info in sidebar
    with st.sidebar.expander("Debug: Environment Settings"):
        st.write(f"Environment: {settings.ENVIRONMENT}")
        st.write(f"Debug Mode: {settings.DEBUG_MODE}")
        st.write(f"DuckDB Path: {settings.DUCKDB_PATH}")
        st.write(f"Query Timeout: {settings.QUERY_TIMEOUT}s")
        
        # Show sensitive settings as masked values
        api_key = settings.OPENAI_API_KEY
        masked_key = "••••" + api_key[-4:] if api_key and len(api_key) > 4 else "Not set"
        st.write(f"OpenAI API Key: {masked_key}") 