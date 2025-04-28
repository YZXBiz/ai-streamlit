"""Tests for DataAnalysisService port."""

import pytest
from app.ports.llm import DataAnalysisService


def test_dataanalysis_service_is_abstract():
    """Test that DataAnalysisService is an abstract base class."""
    # Trying to instantiate DataAnalysisService should raise TypeError
    with pytest.raises(TypeError):
        DataAnalysisService()


def test_dataanalysis_service_abstract_methods():
    """Test that DataAnalysisService has the expected abstract methods."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(DataAnalysisService)
        if getattr(getattr(DataAnalysisService, method_name), "__isabstractmethod__", False)
    ]

    # Check that all the expected methods are abstract
    expected_methods = [
        "load_dataframe",
        "query_dataframe",
        "create_collection",
        "query_collection",
        "get_dataframe_info",
    ]
    for method in expected_methods:
        assert method in abstract_methods


class ConcreteDataAnalysisServiceForTesting(DataAnalysisService):
    """A concrete implementation of DataAnalysisService for testing."""

    async def load_dataframe(self, file_path, name, description=""):
        return {}

    async def query_dataframe(self, query, dataframe_name):
        return "Test response"

    async def create_collection(self, dataframe_names, collection_name, description=""):
        return {}

    async def query_collection(self, query, collection_name):
        return "Test collection response"

    async def get_dataframe_info(self, dataframe_name):
        return {"rows": 10, "columns": 5}


def test_concrete_dataanalysis_service():
    """Test that a concrete implementation of DataAnalysisService can be instantiated."""
    service = ConcreteDataAnalysisServiceForTesting()
    assert isinstance(service, DataAnalysisService)
