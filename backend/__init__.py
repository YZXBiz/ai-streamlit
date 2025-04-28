"""
Backend package for PandasAI Chat Application.

This package provides a FastAPI backend for the PandasAI Chat Application.
"""

__version__ = "1.0.0"

import os

# Define the data_source module for backward compatibility
# This allows imports like `from backend.data_source import CSVSource` to work
import sys
from types import ModuleType

from pandasai import Agent

# Re-export data source classes for backward compatibility
from .app.adapters.db_sources import SQLSource
from .app.adapters.file_sources import CSVSource, ExcelSource, ParquetSource
from .app.ports.datasource import DataSource

data_source = ModuleType("backend.data_source")
data_source.__doc__ = (
    "Data source classes (compatibility module, use backend.app.ports/adapters directly)"
)

# Add all data source classes to the module
data_source.DataSource = DataSource
data_source.CSVSource = CSVSource
data_source.ExcelSource = ExcelSource
data_source.ParquetSource = ParquetSource
data_source.SQLSource = SQLSource

# Register the module
sys.modules["backend.data_source"] = data_source


def create_analyzer():
    """
    Create a PandasAI analyzer instance for Streamlit frontend.

    This function creates and configures a PandasAI Agent with appropriate
    settings for the Streamlit application.

    Returns:
        Agent: Configured PandasAI Agent instance
    """
    # Set up configuration
    config = {
        "display": "streamlit",
        "save_charts": True,
        "save_charts_path": "./charts",
        "enforce_privacy": True,
        "enable_cache": True,
        "max_retries": 3,
        "use_error_correction_framework": True,
    }

    # Create a directory for charts if it doesn't exist
    os.makedirs("./charts", exist_ok=True)

    # Create an empty agent with just configuration
    # We'll add dataframes later when they're loaded
    agent = Agent(
        dfs=[],  # Start with empty dataframes list
        config=config,
        memory_size=10,
    )

    # Add dataframe manager for convenience
    agent.dataframe_manager = {
        "register_dataframe": lambda df, name, description="": agent.add_data(df, name=name)
        if hasattr(agent, "add_data")
        else None
    }

    return agent
