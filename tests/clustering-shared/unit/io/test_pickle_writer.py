"""Tests for pickle writer."""

import os
import pickle
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call

import pytest
import polars as pl
import pandas as pd

from clustering.shared.io.writers.pickle_writer import PickleWriter


class TestPickleWriter:
    """Test suite for PickleWriter."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.test_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })
        self.test_dict = {
            "df1": pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
            "df2": pl.DataFrame({"id": [3, 4], "name": ["Charlie", "Dave"]})
        }

    def test_validate_data_dataframe(self):
        """Test validation with valid DataFrame."""
        writer = PickleWriter(path="test.pkl")
        # This should not raise an exception
        writer._validate_data(self.test_df)

    def test_validate_data_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        writer = PickleWriter(path="test.pkl")
        empty_df = pl.DataFrame({"id": [], "name": []})
        
        with pytest.raises(ValueError, match="Cannot write empty DataFrame"):
            writer._validate_data(empty_df)

    def test_validate_data_dictionary(self):
        """Test validation with valid dictionary of DataFrames."""
        writer = PickleWriter(path="test.pkl")
        # This should not raise an exception
        writer._validate_data(self.test_dict)

    def test_validate_data_empty_dictionary(self):
        """Test validation with empty dictionary."""
        writer = PickleWriter(path="test.pkl")
        empty_dict = {}
        
        with pytest.raises(ValueError, match="Cannot write empty dictionary"):
            writer._validate_data(empty_dict)

    def test_validate_data_dictionary_with_non_dataframe(self):
        """Test validation with dictionary containing non-DataFrame."""
        writer = PickleWriter(path="test.pkl")
        bad_dict = {
            "df1": pl.DataFrame({"id": [1, 2]}),
            "not_df": "This is not a DataFrame"
        }
        
        with pytest.raises(ValueError, match="Dictionary value for key 'not_df' is not a DataFrame"):
            writer._validate_data(bad_dict)

    def test_validate_data_dictionary_with_empty_dataframe(self):
        """Test validation with dictionary containing empty DataFrame."""
        writer = PickleWriter(path="test.pkl")
        bad_dict = {
            "df1": pl.DataFrame({"id": [1, 2]}),
            "empty_df": pl.DataFrame({"id": []})
        }
        
        with pytest.raises(ValueError, match="DataFrame for key 'empty_df' is empty"):
            writer._validate_data(bad_dict)

    def test_validate_data_unsupported_type(self):
        """Test validation with unsupported data type."""
        writer = PickleWriter(path="test.pkl")
        
        with pytest.raises(ValueError, match="Unsupported data type"):
            writer._validate_data("This is not a DataFrame or dictionary")

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_write_to_destination_dataframe(self, mock_pickle_dump, mock_file):
        """Test writing DataFrame to pickle file."""
        # Setup
        writer = PickleWriter(path="test.pkl")
        
        # Execute
        writer._write_to_destination(self.test_df)
        
        # Verify
        mock_file.assert_called_once_with("test.pkl", "wb")
        
        # Check that pickle.dump was called with correct arguments
        # The protocol is passed as a keyword argument not positional
        mock_pickle_dump.assert_called_once()
        assert mock_pickle_dump.call_args[0][0] is not None  # Data to pickle
        assert mock_pickle_dump.call_args[0][1] == mock_file.return_value  # File handle
        assert mock_pickle_dump.call_args[1]['protocol'] == pickle.HIGHEST_PROTOCOL  # Protocol as kwarg
        
        # Verify that the DataFrame was converted to pandas
        assert isinstance(mock_pickle_dump.call_args[0][0], pd.DataFrame)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_write_to_destination_dictionary(self, mock_pickle_dump, mock_file):
        """Test writing dictionary of DataFrames to pickle file."""
        # Setup
        writer = PickleWriter(path="test.pkl")
        
        # Execute
        writer._write_to_destination(self.test_dict)
        
        # Verify
        mock_file.assert_called_once_with("test.pkl", "wb")
        
        # Check that pickle.dump was called with correct arguments
        mock_pickle_dump.assert_called_once()
        assert mock_pickle_dump.call_args[0][0] is not None  # Data to pickle
        assert mock_pickle_dump.call_args[0][1] == mock_file.return_value  # File handle
        assert mock_pickle_dump.call_args[1]['protocol'] == pickle.HIGHEST_PROTOCOL  # Protocol as kwarg
        
        # Verify that the dictionary values were converted to pandas DataFrames
        pickled_data = mock_pickle_dump.call_args[0][0]
        assert isinstance(pickled_data, dict)
        assert len(pickled_data) == len(self.test_dict)
        assert all(isinstance(df, pd.DataFrame) for df in pickled_data.values())

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_custom_protocol(self, mock_pickle_dump, mock_file):
        """Test writing with custom protocol."""
        # Setup
        writer = PickleWriter(path="test.pkl", protocol=4)
        
        # Execute
        writer._write_to_destination(self.test_df)
        
        # Verify
        mock_file.assert_called_once_with("test.pkl", "wb")
        
        # Check that pickle.dump was called with protocol 4
        mock_pickle_dump.assert_called_once()
        assert mock_pickle_dump.call_args[1]['protocol'] == 4

    def test_integration_with_writer_base_class(self):
        """Test integration with the Writer base class."""
        # Create a temporary file to write to
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Setup
            writer = PickleWriter(path=tmp_path)
            
            # Execute - this calls the template method in Writer
            writer.write(self.test_df)
            
            # Verify that the file exists and can be read back
            assert os.path.exists(tmp_path)
            
            # Read the file back and verify contents
            with open(tmp_path, "rb") as f:
                data = pickle.load(f)
                assert isinstance(data, pd.DataFrame)
                assert len(data) == len(self.test_df)
                assert list(data.columns) == list(self.test_df.columns)
        
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path) 