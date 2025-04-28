"""Utility functions for testing PandasAI functionality.

This module provides mock implementations and utilities for testing
PandasAI interactions without requiring actual LLM API calls.
"""

from unittest.mock import MagicMock, patch

import pandas as pd


# Sample dataframes for testing
def get_test_dataframe() -> pd.DataFrame:
    """Get a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03"],
            "region": ["North", "South", "East", "West", "North"],
            "product": ["Widget A", "Widget B", "Widget A", "Widget B", "Widget C"],
            "sales": [1200, 950, 1350, 1100, 800],
        }
    )


def get_sample_df_for_charts() -> pd.DataFrame:
    """Get a sample DataFrame suitable for chart generation tests."""
    return pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
            "sales": [1200, 1400, 1100, 1600, 1800],
            "region": ["North", "North", "South", "South", "East"],
        }
    )


class MockAgent:
    """Mock implementation of PandasAI Agent for testing."""

    def __init__(self, dfs=None, config=None, memory_size=5, description=None, vectorstore=None):
        """Initialize the mock agent."""
        self.dfs = dfs or []
        self.config = config or {}
        self.memory_size = memory_size
        self.description = description
        self.vectorstore = vectorstore
        self.history = []

    def chat(self, question: str) -> str:
        """Mock implementation of chat method."""
        self.history = [(question, self._generate_response(question))]
        return self.history[-1][1]

    def follow_up(self, question: str) -> str:
        """Mock implementation of follow_up method."""
        response = self._generate_response(question, follow_up=True)
        self.history.append((question, response))
        return response

    def _generate_response(self, question: str, follow_up=False) -> str:
        """Generate a mock response based on the question."""
        question_lower = question.lower()

        # For testing charts
        if "plot" in question_lower or "chart" in question_lower or "graph" in question_lower:
            return "Here's a chart showing the data you requested. [CHART_DATA]"

        # For testing total calculations
        if "total" in question_lower and "sales" in question_lower:
            if "region" in question_lower:
                return "Total sales by region: North: 2000, South: 950, East: 1350, West: 1100"
            return "The total sales are 5400."

        # For testing filtering
        if "highest" in question_lower or "maximum" in question_lower:
            return "Widget A has the highest sales with a total of 2550."

        # For testing breakdowns
        if "break" in question_lower and "down" in question_lower:
            if "region" in question_lower:
                return "Sales breakdown by region: North: 2000 (37%), South: 950 (18%), East: 1350 (25%), West: 1100 (20%)"
            if "product" in question_lower:
                return "Sales breakdown by product: Widget A: 2550 (47%), Widget B: 2050 (38%), Widget C: 800 (15%)"

        # Default response for questions not matching patterns
        if follow_up:
            return "Based on our previous discussion, I can tell you that the data shows interesting patterns."
        return "Analysis of the data shows some interesting patterns. Can you be more specific about what you'd like to know?"


def create_mock_agent():
    """Create a mock PandasAI Agent for testing."""
    return MockAgent(dfs=[get_test_dataframe()])


@patch("pandasai.Agent")
def mock_pandasai_agent(mock_agent_class):
    """Patch the PandasAI Agent class with our mock implementation."""
    mock_instance = create_mock_agent()
    mock_agent_class.return_value = mock_instance
    return mock_instance
