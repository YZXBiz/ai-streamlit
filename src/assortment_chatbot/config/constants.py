"""
Configuration constants for the assortment_chatbot application.

This module contains configuration constants and settings used across
the assortment_chatbot application, organized by component.
"""

import os
from typing import Any, TypedDict

# Data configuration settings
DATA_CONFIG: dict[str, Any] = {
    "allowed_extensions": [".csv", ".xlsx", ".xls", ".json"],
    "max_file_size_mb": 50,
    "max_rows": 100000,
    "handle_missing_values": True,
    "missing_values_strategy": "fill",  # Options: "drop", "fill"
    "fill_value": 0,
    "enable_download": True,
    "csv_encoding": "utf-8",
}

# Visualization configuration settings
VIZ_CONFIG: dict[str, Any] = {
    "default_chart_type": "bar",
    "color_palette": "viridis",
    "enable_pygwalker": True,
    "max_categories": 20,
    "chart_height": 400,
    "chart_width": 700,
}

# Clustering configuration settings
CLUSTERING_CONFIG: dict[str, Any] = {
    "default_algorithm": "kmeans",
    "available_algorithms": ["kmeans", "dbscan", "hierarchical"],
    "default_n_clusters": 3,
    "max_clusters": 10,
    "random_state": 42,
}

# AI configuration settings
AI_CONFIG: dict[str, Any] = {
    "enable_ai_features": True,
    "default_model": "openai",
    "api_keys": {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    },
    "max_tokens": 1000,
    "temperature": 0.7,
}

# UI configuration settings
UI_CONFIG: dict[str, Any] = {
    "page_title": "Data Analytics Dashboard",
    "sidebar_width": 300,
    "theme": "light",  # Options: "light", "dark"
    "logo_path": "assets/logo.png",
    "enable_custom_css": True,
}

# Database configuration settings
DB_CONFIG: dict[str, Any] = {
    "use_duckdb": True,
    "snowflake": {
        "enabled": False,
        "account": os.getenv("SNOWFLAKE_ACCOUNT", ""),
        "user": os.getenv("SNOWFLAKE_USER", ""),
        "password": os.getenv("SNOWFLAKE_PASSWORD", ""),
        "database": os.getenv("SNOWFLAKE_DATABASE", ""),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", ""),
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", ""),
    },
}


class NavItem(TypedDict):
    """Navigation item configuration."""

    icon: str
    index: int


class PageConfig(TypedDict):
    """Page configuration settings."""

    page_title: str
    page_icon: str
    layout: str
    initial_sidebar_state: str


class DataConfig(TypedDict):
    """Data handling configuration."""

    allowed_extensions: list[str]
    max_file_size_mb: int
    preview_rows: int
    supported_encodings: list[str]


# Application version and metadata
APP_VERSION = "v0.2.0"
APP_DESCRIPTION = "Chat with your data using PydanticAI"
DEBUG_MODE = False

# Feature flags
FEATURES = {
    "enable_clustering": True,
    "enable_data_export": True,
    "enable_advanced_viz": True,
    "enable_ai_chat": True,
}

# Data handling configuration
DATA_CONFIG: DataConfig = {
    "allowed_extensions": [".csv", ".xlsx", ".json"],
    "max_file_size_mb": 100,
    "preview_rows": 1000,
    "supported_encodings": ["utf-8", "latin1", "iso-8859-1"],
}

# Page configuration
PAGE_CONFIG: PageConfig = {
    "page_title": "Data Chat Assistant",
    "page_icon": "🤖",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Navigation configuration
NAV_ITEMS: dict[str, NavItem] = {
    "Home": {"icon": "house", "index": 0},
    "Data Uploader": {"icon": "upload", "index": 1},
    "Interactive Visualization": {"icon": "bar-chart-fill", "index": 2},
    "Cluster Analysis": {"icon": "bar-chart", "index": 3},
    "AI Chat": {"icon": "robot", "index": 4},
}

# Theme configuration
THEME = {
    "primary_color": "#FF4B4B",
    "background_color": "#FFFFFF",
    "secondary_background_color": "#F0F2F6",
    "text_color": "#262730",
}

# CSS Styles
CUSTOM_CSS = f"""
<style>
.main .block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    background-color: {THEME["background_color"]};
}}
.stAlert > div {{
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}}
.stApp {{
    color: {THEME["text_color"]};
}}
.stButton>button {{
    background-color: {THEME["primary_color"]};
    color: white;
}}
</style>
"""
