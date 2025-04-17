"""Tests for clustering.io.readers.csv_reader module."""

import os
import tempfile

import pandas as pd
import polars as pl
import pytest

from clustering.io.readers.csv_reader import CSVReader


class TestCSVReader:
    """Tests for CSVReader class."""

    @pytest.fixture
    def sample_csv_file(self) -> str:
        """Create a temporary CSV file for testing.

        Returns:
            Path to the temporary CSV file
        """
        # Create a temporary CSV file
        data = {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
            "value": [10.5, 20.3, 30.1, 40.8, 50.2],
            "date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],
        }

        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as temp:
            pd.DataFrame(data).to_csv(temp.name, index=False)
            temp_path = temp.name

        yield temp_path

        # Clean up after test
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_read_csv_file(self, sample_csv_file) -> None:
        """Test reading a standard CSV file."""
        reader = CSVReader(path=sample_csv_file)
        result = reader.read()

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (5, 4)
        assert list(result.columns) == ["id", "name", "value", "date"]
        assert result["id"][0] == 1
        assert result["name"][1] == "Bob"
        assert result["value"][2] == 30.1

    def test_custom_delimiter(self, sample_csv_file) -> None:
        """Test reading a CSV with custom delimiter."""
        # Create a TSV file
        data = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.3, 30.1]}

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w+", delete=False) as temp:
            pd.DataFrame(data).to_csv(temp.name, sep="\t", index=False)
            temp_path = temp.name

        try:
            reader = CSVReader(path=temp_path, delimiter="\t")
            result = reader.read()

            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert list(result.columns) == ["id", "name", "value"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_no_header(self) -> None:
        """Test reading a CSV file without header."""
        # Create a CSV without header
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as temp:
            temp.write(b"1,Alice,10.5\n2,Bob,20.3\n3,Charlie,30.1\n")
            temp_path = temp.name

        try:
            reader = CSVReader(path=temp_path, has_header=False)
            result = reader.read()

            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            # Default column names should be column_1, column_2, column_3
            assert all(c.startswith("column_") for c in result.columns)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_limit_rows(self, sample_csv_file) -> None:
        """Test limiting the number of rows read."""
        reader = CSVReader(path=sample_csv_file, limit=2)
        result = reader.read()

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 4)

    def test_null_values(self) -> None:
        """Test handling of null values."""
        # Create a CSV with some null values
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as temp:
            temp.write(b"id,name,value\n1,Alice,10.5\n2,,20.3\n3,Charlie,\n")
            temp_path = temp.name

        try:
            reader = CSVReader(path=temp_path, null_values=[""])
            result = reader.read()

            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert result["name"][1] is None
            assert result["value"][2] is None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_fallback_to_pandas(self, mocker) -> None:
        """Test fallback to pandas when polars reader fails."""
        # Create a sample CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as temp:
            temp.write(b"id,name,value\n1,Alice,10.5\n2,Bob,20.3\n3,Charlie,30.1\n")
            temp_path = temp.name

        try:
            # Mock polars.read_csv to raise an exception, forcing fallback to pandas
            mocker.patch("polars.read_csv", side_effect=Exception("Simulated polars error"))

            reader = CSVReader(path=temp_path)
            result = reader.read()

            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert list(result.columns) == ["id", "name", "value"]
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
