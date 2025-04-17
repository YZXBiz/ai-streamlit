"""Tests for Dagster data I/O resources in the clustering pipeline."""

import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest
from dagster import build_init_resource_context, build_op_context

from clustering.dagster.resources.data_io import data_reader, data_writer


@pytest.fixture
def test_data() -> pd.DataFrame:
    """Create test data for I/O operations.

    Returns:
        DataFrame with test data
    """
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["A", "B", "C", "D", "E"],
            "value": [10.1, 20.2, 30.3, 40.4, 50.5],
        }
    )


@pytest.fixture
def temp_csv_file() -> Generator[Path, None, None]:
    """Create a temporary CSV file for testing.

    Yields:
        Path to the temporary CSV file
    """
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)

        # Create test data and save to file
        test_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["A", "B", "C", "D", "E"],
                "value": [10.1, 20.2, 30.3, 40.4, 50.5],
            }
        )
        test_df.to_csv(temp_path, index=False)

    yield temp_path

    # Clean up
    if temp_path.exists():
        os.unlink(temp_path)


class TestDataReader:
    """Tests for the data_reader resource."""

    def test_csv_reader(self, temp_csv_file) -> None:
        """Test reading from a CSV file."""
        # Create context with config for CSV reader
        init_context = build_init_resource_context(
            config={"type": "csv", "path": str(temp_csv_file), "options": {"index_col": None}}
        )

        # Initialize the resource
        reader = data_reader(init_context)

        # Read the data
        result = reader()

        # Verify the result
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (5, 3)
        assert list(result.columns) == ["id", "name", "value"]
        assert result["id"].tolist() == [1, 2, 3, 4, 5]

    def test_reader_with_filter(self, temp_csv_file) -> None:
        """Test reading with a filter function."""
        # Create context with config including a filter
        init_context = build_init_resource_context(
            config={
                "type": "csv",
                "path": str(temp_csv_file),
                "options": {},
                "filter": "value > 30",
            }
        )

        # Initialize the resource
        reader = data_reader(init_context)

        # Read the data
        result = reader()

        # Verify the result is filtered
        assert result.shape[0] == 2  # Only two rows should match the filter
        assert all(val > 30 for val in result["value"])

    def test_reader_in_op_context(self, temp_csv_file) -> None:
        """Test using the reader in an op context."""
        # Create reader resource
        init_context = build_init_resource_context(
            config={"type": "csv", "path": str(temp_csv_file), "options": {}}
        )
        reader_instance = data_reader(init_context)

        # Build op context with the reader resource
        op_context = build_op_context(resources={"test_reader": reader_instance})

        # Use the reader from context
        data = op_context.resources.test_reader()

        # Verify data was read correctly
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == 5

    @pytest.mark.parametrize(
        "bad_config,expected_error",
        [
            ({"type": "unknown"}, "Unsupported reader type"),
            ({"type": "csv"}, "Missing required configuration"),
            ({}, "Missing required configuration"),
        ],
    )
    def test_reader_validation(self, bad_config, expected_error) -> None:
        """Test validation of reader configuration.

        Args:
            bad_config: Invalid configuration to test
            expected_error: Expected error message pattern
        """
        # Create context with invalid config
        init_context = build_init_resource_context(config=bad_config)

        # Initialize should raise an error
        with pytest.raises(Exception, match=expected_error):
            reader = data_reader(init_context)
            reader()


class TestDataWriter:
    """Tests for the data_writer resource."""

    def test_csv_writer(self, test_data) -> None:
        """Test writing to a CSV file."""
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            output_path = Path(temp.name)

        try:
            # Create context with config for CSV writer
            init_context = build_init_resource_context(
                config={"type": "csv", "path": str(output_path), "options": {"index": False}}
            )

            # Initialize the resource
            writer = data_writer(init_context)

            # Write the data
            result = writer(test_data)

            # Verify the result
            assert result == len(test_data)
            assert output_path.exists()

            # Read back and verify
            read_back = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(read_back, test_data)
        finally:
            # Clean up
            if output_path.exists():
                os.unlink(output_path)

    def test_writer_with_transform(self, test_data) -> None:
        """Test writing with a transform function."""
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            output_path = Path(temp.name)

        try:
            # Create context with config including a transform
            init_context = build_init_resource_context(
                config={
                    "type": "csv",
                    "path": str(output_path),
                    "options": {"index": False},
                    "transform": "value = value * 2",
                }
            )

            # Initialize the resource
            writer = data_writer(init_context)

            # Write the data
            writer(test_data.copy())

            # Read back and verify the transform was applied
            read_back = pd.read_csv(output_path)
            expected = test_data.copy()
            expected["value"] = expected["value"] * 2
            pd.testing.assert_frame_equal(read_back, expected)
        finally:
            # Clean up
            if output_path.exists():
                os.unlink(output_path)

    def test_writer_in_op_context(self, test_data) -> None:
        """Test using the writer in an op context."""
        # Create a temporary output file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            output_path = Path(temp.name)

        try:
            # Create writer resource
            init_context = build_init_resource_context(
                config={"type": "csv", "path": str(output_path), "options": {"index": False}}
            )
            writer_instance = data_writer(init_context)

            # Build op context with the writer resource
            op_context = build_op_context(resources={"test_writer": writer_instance})

            # Use the writer from context
            result = op_context.resources.test_writer(test_data)

            # Verify data was written correctly
            assert result == len(test_data)
            assert output_path.exists()
            read_back = pd.read_csv(output_path)
            pd.testing.assert_frame_equal(read_back, test_data)
        finally:
            # Clean up
            if output_path.exists():
                os.unlink(output_path)


@pytest.mark.integration
def test_reader_writer_integration(test_data) -> None:
    """Integration test for reader and writer resources together."""
    # Create a temporary file for the round-trip test
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # First write data to the file
        write_context = build_init_resource_context(
            config={"type": "csv", "path": str(temp_path), "options": {"index": False}}
        )
        writer = data_writer(write_context)
        writer(test_data)

        # Then read it back
        read_context = build_init_resource_context(
            config={"type": "csv", "path": str(temp_path), "options": {}}
        )
        reader = data_reader(read_context)
        read_data = reader()

        # Verify round-trip integrity
        pd.testing.assert_frame_equal(read_data, test_data)
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)
