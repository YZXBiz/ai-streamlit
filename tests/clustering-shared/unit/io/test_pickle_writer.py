"""Tests for the PickleWriter class."""

import os
import pickle
from pathlib import Path
from unittest import mock

import polars as pl
import pytest

from clustering.shared.io.writers.pickle_writer import PickleWriter


class TestPickleWriter:
    """Tests for PickleWriter."""

    def setup_method(self):
        """Set up before each test."""
        self.temp_file = "test_data.pkl"
        # Ensure the file doesn't exist before tests
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_validate_data_dataframe(self):
        """Test validation with a valid DataFrame."""
        writer = PickleWriter(path=self.temp_file)
        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Should not raise an exception
        writer._validate_data(df)

    def test_validate_data_empty_dataframe(self):
        """Test validation with an empty DataFrame."""
        writer = PickleWriter(path=self.temp_file)
        df = pl.DataFrame({"col1": [], "col2": []})

        with pytest.raises(ValueError, match="Cannot write empty DataFrame"):
            writer._validate_data(df)

    def test_validate_data_dict_of_dataframes(self):
        """Test validation with a dictionary of DataFrames."""
        writer = PickleWriter(path=self.temp_file)
        data = {
            "df1": pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            "df2": pl.DataFrame({"col3": [4, 5, 6], "col4": ["d", "e", "f"]}),
        }

        # Should not raise an exception
        writer._validate_data(data)

    def test_validate_data_empty_dict(self):
        """Test validation with an empty dictionary."""
        writer = PickleWriter(path=self.temp_file)
        data = {}

        with pytest.raises(ValueError, match="Cannot write empty dictionary"):
            writer._validate_data(data)

    def test_validate_data_dict_with_non_dataframe(self):
        """Test validation with a dictionary containing a non-DataFrame value."""
        writer = PickleWriter(path=self.temp_file)
        data = {
            "df1": pl.DataFrame({"col1": [1, 2, 3]}),
            "not_df": "This is not a DataFrame",
        }

        with pytest.raises(ValueError, match="not a DataFrame"):
            writer._validate_data(data)

    def test_validate_data_dict_with_empty_dataframe(self):
        """Test validation with a dictionary containing an empty DataFrame."""
        writer = PickleWriter(path=self.temp_file)
        data = {
            "df1": pl.DataFrame({"col1": [1, 2, 3]}),
            "empty_df": pl.DataFrame({"col2": []}),
        }

        with pytest.raises(ValueError, match="DataFrame for key 'empty_df' is empty"):
            writer._validate_data(data)

    def test_validate_data_unsupported_type(self):
        """Test validation with an unsupported data type."""
        writer = PickleWriter(path=self.temp_file)
        data = "This is not a DataFrame or dictionary"

        with pytest.raises(ValueError, match="Unsupported data type"):
            writer._validate_data(data)

    def test_write_to_destination_dataframe(self):
        """Test writing a DataFrame to a pickle file."""
        writer = PickleWriter(path=self.temp_file)
        df = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Write the data
        writer._write_to_destination(df)

        # Check that the file exists
        assert os.path.exists(self.temp_file)

        # Read the file back and verify the content
        with open(self.temp_file, "rb") as f:
            loaded_data = pickle.load(f)

        # Compare with pandas DataFrame (since we convert to pandas before saving)
        pd_df = df.to_pandas()
        assert loaded_data.equals(pd_df)

    def test_write_to_destination_dict(self):
        """Test writing a dictionary of DataFrames to a pickle file."""
        writer = PickleWriter(path=self.temp_file)
        data = {
            "df1": pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]}),
            "df2": pl.DataFrame({"col3": [4, 5, 6], "col4": ["d", "e", "f"]}),
        }

        # Write the data
        writer._write_to_destination(data)

        # Check that the file exists
        assert os.path.exists(self.temp_file)

        # Read the file back and verify the content
        with open(self.temp_file, "rb") as f:
            loaded_data = pickle.load(f)

        # Verify the dictionary structure
        assert isinstance(loaded_data, dict)
        assert set(loaded_data.keys()) == set(data.keys())

        # Compare each DataFrame (converted to pandas)
        for key, df in data.items():
            pd_df = df.to_pandas()
            assert loaded_data[key].equals(pd_df)

    def test_protocol_parameter(self):
        """Test using a custom protocol parameter."""
        writer = PickleWriter(path=self.temp_file, protocol=2)
        df = pl.DataFrame({"col1": [1, 2, 3]})

        # Mock the pickle.dump function to check that it's called with the correct protocol
        with mock.patch("pickle.dump") as mock_dump:
            writer._write_to_destination(df)

            # Verify that pickle.dump was called with protocol=2
            args, kwargs = mock_dump.call_args
            assert kwargs["protocol"] == 2
