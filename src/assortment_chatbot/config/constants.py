"""
Configuration constants for the assortment_chatbot application.

This module contains configuration constants and settings used across
the assortment_chatbot application, organized by component.
"""

import os
from typing import Any, TypedDict


class NavItem(TypedDict):
    """Navigation item configuration."""

    icon: str


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
    "max_rows": 100000,
    "handle_missing_values": True,
    "missing_values_strategy": "fill",  # Options: "drop", "fill"
    "fill_value": 0,
    "enable_download": True,
    "csv_encoding": "utf-8",
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
    "Home": {"icon": "house"},
    "Data Uploader": {"icon": "upload"},
    "AI Chat": {"icon": "robot"},
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
