"""
Backend package for PandasAI Chat Application.

This package provides a FastAPI backend for the PandasAI Chat Application.
"""

__version__ = "1.0.0"

import os

from pandasai import Agent


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
