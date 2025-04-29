"""Tests for the analyzer service.

This module contains tests for the AnalyzerService which
integrates with PandasAI for data analysis.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from backend.app.services.analyzer_service import AnalyzerService
from backend.app.services.dataframe_service import DataFrameService
from fastapi import UploadFile

from ..utils.test_pandasai_utils import MockAgent, get_test_dataframe


@pytest.fixture
def test_csv_path():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df = get_test_dataframe()
        df.to_csv(tmp.name, index=False)
        yield tmp.name
    # Cleanup
    os.unlink(tmp.name)


@pytest.fixture
def mock_dataframe_service():
    """Create a mock DataFrameService."""
    service = MagicMock(spec=DataFrameService)
    service.load_dataframe.return_value = get_test_dataframe()
    return service


@pytest.fixture
@patch("pandasai.Agent")
def analyzer_service(mock_agent_class, mock_dataframe_service):
    """Create an AnalyzerService with mocked dependencies."""
    # Setup mock agent
    mock_agent = MockAgent()
    mock_agent_class.return_value = mock_agent

    # Create service
    service = AnalyzerService(dataframe_service=mock_dataframe_service)
    return service, mock_agent


@pytest.mark.asyncio
async def test_initialize_agent_for_session(analyzer_service, test_csv_path):
    """Test initializing a PandasAI agent for a session."""
    service, mock_agent = analyzer_service

    # Create a mock session
    session_id = 1
    file_path = test_csv_path

    # Initialize agent
    agent = await service.initialize_agent_for_session(session_id, file_path)

    # Verify the agent was created with expected parameters
    assert agent is not None
    assert agent is mock_agent


@pytest.mark.asyncio
async def test_ask_question_new_session(analyzer_service, test_csv_path):
    """Test asking a question in a new session."""
    service, mock_agent = analyzer_service

    # Setup
    session_id = 1
    file_path = test_csv_path
    question = "What is the total sales?"

    # Ask the question
    answer = await service.ask_question(session_id, file_path, question)

    # Verify
    assert answer is not None
    assert isinstance(answer, str)
    assert "total sales" in answer.lower()
    # Should use chat() for first question
    assert mock_agent.history


@pytest.mark.asyncio
async def test_ask_follow_up_question(analyzer_service, test_csv_path):
    """Test asking follow-up questions."""
    service, mock_agent = analyzer_service

    # Setup - First question
    session_id = 1
    file_path = test_csv_path
    first_question = "What is the total sales?"
    await service.ask_question(session_id, file_path, first_question)

    # Follow-up question
    follow_up = "Break it down by region"
    answer = await service.ask_question(session_id, file_path, follow_up)

    # Verify
    assert answer is not None
    assert isinstance(answer, str)
    assert "breakdown by region" in answer.lower()
    # Should have at least 2 items in history
    assert len(mock_agent.history) >= 2


@pytest.mark.asyncio
async def test_handle_chart_generation(analyzer_service, test_csv_path):
    """Test handling chart generation."""
    service, mock_agent = analyzer_service

    # Setup
    session_id = 1
    file_path = test_csv_path
    question = "Plot the sales trend"

    # Mock the response to indicate a chart was generated
    mock_agent.chat = MagicMock(
        return_value="Here's a chart showing the data you requested. [CHART_DATA]"
    )

    # Ask the question
    answer = await service.ask_question(session_id, file_path, question)

    # Verify
    assert answer is not None
    assert "chart" in answer.lower()
    assert "[CHART_DATA]" in answer


@pytest.mark.asyncio
async def test_agent_reuse(analyzer_service, test_csv_path):
    """Test that agents are reused for the same session."""
    service, _ = analyzer_service

    # Setup
    session_id = 1
    file_path = test_csv_path

    # Initialize agent twice for the same session
    agent1 = await service.initialize_agent_for_session(session_id, file_path)
    agent2 = await service.initialize_agent_for_session(session_id, file_path)

    # Should be the same agent instance
    assert agent1 is agent2

    # Different session should get a different agent
    another_session_id = 2
    agent3 = await service.initialize_agent_for_session(another_session_id, file_path)

    # Should not be the same agent instance
    assert agent1 is not agent3


@pytest.mark.asyncio
async def test_error_handling(analyzer_service, test_csv_path):
    """Test error handling in the analyzer service."""
    service, mock_agent = analyzer_service

    # Setup
    session_id = 1
    file_path = test_csv_path

    # Mock chat method to raise an exception
    mock_agent.chat = MagicMock(side_effect=Exception("Test error"))

    # Ask a question that will cause an error
    question = "This will cause an error"

    # Should handle the error and return an error message
    answer = await service.ask_question(session_id, file_path, question)

    assert answer is not None
    assert "error" in answer.lower() or "unable" in answer.lower()


@pytest.mark.asyncio
async def test_service_close(analyzer_service):
    """Test closing the analyzer service."""
    service, _ = analyzer_service

    # Should not raise any exceptions
    await service.close()
    # Calling close multiple times should be safe
    await service.close()
