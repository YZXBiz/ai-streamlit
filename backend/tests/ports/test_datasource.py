"""Tests for datasource module."""

import inspect
from abc import ABC
from typing import Optional

import pytest


def test_datasource_constants():
    """Test datasource module constants."""
    from app.ports.datasource import DataSource

    assert hasattr(DataSource, "__abstractmethods__")


class TestDataSource:
    """Tests for the DataSource interface."""

    @pytest.mark.port
    def test_data_source_is_abc(self):
        """Test that DataSource is an ABC."""
        from app.ports.datasource import DataSource

        assert issubclass(DataSource, ABC)

    @pytest.mark.port
    def test_data_source_methods(self):
        """Test that DataSource has the expected methods."""
        from app.ports.datasource import DataSource

        # Check required methods
        assert hasattr(DataSource, "load")
        assert hasattr(DataSource, "__init__")

    @pytest.mark.port
    def test_data_source_init_signature(self):
        """Test that DataSource.__init__ has the expected signature."""
        from app.ports.datasource import DataSource

        sig = inspect.signature(DataSource.__init__)
        params = sig.parameters

        assert "self" in params
        assert "source" in params
        assert "name" in params
        assert "description" in params

        # Check parameter types and defaults
        assert params["description"].annotation == "Optional[str]"
        assert params["description"].default is None

    @pytest.mark.port
    def test_data_source_init_functionality(self):
        """Test that DataSource.__init__ sets attributes correctly."""
        from app.ports.datasource import DataSource

        # Create a concrete subclass for testing
        class TestConcreteDataSource(DataSource):
            def load(self):
                pass

        # Test with all parameters
        source = TestConcreteDataSource(
            source="test_source", name="test_name", description="test_description"
        )
        assert source.source == "test_source"
        assert source.name == "test_name"
        assert source.description == "test_description"

        # Test with default description
        source = TestConcreteDataSource(source="test_source", name="test_name")
        assert source.source == "test_source"
        assert source.name == "test_name"
        assert source.description == "Data from test_source"

    @pytest.mark.port
    def test_data_source_load_signature(self):
        """Test that DataSource.load has the expected signature."""
        import pandasai as pai

        from app.ports.datasource import DataSource

        sig = inspect.signature(DataSource.load)
        params = sig.parameters

        assert "self" in params

        # Check return type
        assert sig.return_annotation == pai.DataFrame
