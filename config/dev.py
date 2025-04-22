"""
Development environment configuration.

This module contains configuration settings for the development environment.
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Environment
ENV = "development"

# MongoDB
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/chatbot")

# DuckDB
DUCKDB_PATH = os.environ.get("DUCKDB_PATH", ":memory:")

# Streamlit
STREAMLIT_SERVER_PORT = int(os.environ.get("STREAMLIT_SERVER_PORT", 8501))
STREAMLIT_THEME_BASE = os.environ.get("STREAMLIT_THEME_BASE", "light")

# Sample data
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_data.csv"

# Debug
DEBUG = os.environ.get("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# Snowflake (for development, optional)
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "")
SNOWFLAKE_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA", "") 