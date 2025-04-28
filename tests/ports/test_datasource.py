"""Tests for DataSource port."""

import pytest
from app.ports.datasource import DataSource


def test_datasource_is_abstract():
    """Test that DataSource is an abstract base class."""
    # Trying to instantiate DataSource should raise TypeError
    with pytest.raises(TypeError):
        DataSource(source="test", name="test")


def test_datasource_abstract_methods():
    """Test that DataSource has the expected abstract methods."""
    # Check that the expected methods are abstract
    abstract_methods = [
        method_name
        for method_name in dir(DataSource)
        if getattr(getattr(DataSource, method_name), "__isabstractmethod__", False)
    ]

    # Check that all the expected methods are abstract
    expected_methods = ["load"]
    for method in expected_methods:
        assert method in abstract_methods


class ConcreteDataSourceForTesting(DataSource):
    """A concrete implementation of DataSource for testing."""

    def load(self):
        return {}


def test_concrete_datasource():
    """Test that a concrete implementation of DataSource can be instantiated."""
    source = ConcreteDataSourceForTesting(source="test", name="test")
    assert isinstance(source, DataSource)
    assert source.source == "test"
    assert source.name == "test"
    assert source.description == "Data from test"
