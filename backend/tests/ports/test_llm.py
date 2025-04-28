"""Tests for LLM module."""

from abc import ABC
from typing import Protocol

import pytest


def test_llm_constants():
    """Test LLM module constants."""
    from app.ports.llm import DataAnalysisService

    assert hasattr(DataAnalysisService, "__abstractmethods__")


class TestDataAnalysisService:
    """Tests for the DataAnalysisService interface."""

    @pytest.mark.port
    def test_data_analysis_service_is_abc(self):
        """Test that DataAnalysisService is an ABC."""
        from app.ports.llm import DataAnalysisService

        assert issubclass(DataAnalysisService, ABC)

    @pytest.mark.port
    def test_data_analysis_service_methods(self):
        """Test that DataAnalysisService has the expected methods."""
        from app.ports.llm import DataAnalysisService

        # Check required methods
        assert hasattr(DataAnalysisService, "load_dataframe")
        assert hasattr(DataAnalysisService, "query_dataframe")
        assert hasattr(DataAnalysisService, "create_collection")
        assert hasattr(DataAnalysisService, "query_collection")
        assert hasattr(DataAnalysisService, "get_dataframe_info")

    @pytest.mark.port
    def test_data_analysis_service_method_signatures(self):
        """Test that DataAnalysisService methods have the expected signatures."""
        import inspect

        from app.ports.llm import DataAnalysisService

        # Check load_dataframe method
        sig = inspect.signature(DataAnalysisService.load_dataframe)
        params = sig.parameters
        assert "self" in params
        assert "file_path" in params
        assert "name" in params
        assert "description" in params
        assert params["description"].default == ""

        # Check query_dataframe method
        sig = inspect.signature(DataAnalysisService.query_dataframe)
        params = sig.parameters
        assert "self" in params
        assert "query" in params
        assert "dataframe_name" in params

        # Check create_collection method
        sig = inspect.signature(DataAnalysisService.create_collection)
        params = sig.parameters
        assert "self" in params
        assert "dataframe_names" in params
        assert "collection_name" in params
        assert "description" in params
        assert params["description"].default == ""

        # Check query_collection method
        sig = inspect.signature(DataAnalysisService.query_collection)
        params = sig.parameters
        assert "self" in params
        assert "query" in params
        assert "collection_name" in params

        # Check get_dataframe_info method
        sig = inspect.signature(DataAnalysisService.get_dataframe_info)
        params = sig.parameters
        assert "self" in params
        assert "dataframe_name" in params
