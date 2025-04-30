import os
from unittest import mock

import pandas as pd
import pytest
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.models.data_model import DataModel


@pytest.fixture
def mock_file():
    """Create a mock uploaded file."""
    file = mock.Mock(spec=UploadedFile)
    file.name = "test_data.csv"
    file.getvalue.return_value = b"col1,col2\n1,a\n2,b"
    return file


@pytest.fixture
def mock_duplicate_files():
    """Create mock uploaded files with duplicate names."""
    file1 = mock.Mock(spec=UploadedFile)
    file1.name = "data.csv"
    file1.getvalue.return_value = b"col1,col2\n1,a\n2,b"
    
    file2 = mock.Mock(spec=UploadedFile)
    file2.name = "data.csv"  # Same name as file1
    file2.getvalue.return_value = b"col3,col4\n3,c\n4,d"
    
    file3 = mock.Mock(spec=UploadedFile)
    file3.name = "data.xlsx"  # Same base name, different extension
    file3.getvalue.return_value = b"col5,col6\n5,e\n6,f"
    
    return [file1, file2, file3]


def test_load_dataframe():
    """Test loading a single dataframe."""
    # Create a mock file
    file = mock.Mock(spec=UploadedFile)
    file.name = "test.csv"
    file.getvalue.return_value = b"a,b\n1,2\n3,4"
    
    # Mock temporary file and pandas read_csv
    with mock.patch("tempfile.NamedTemporaryFile") as mock_tmp:
        mock_tmp_file = mock.MagicMock()
        mock_tmp_file.__enter__.return_value = mock_tmp_file
        mock_tmp_file.name = "tmp_file_path"
        mock_tmp.return_value = mock_tmp_file
        
        with mock.patch("pandas.read_csv") as mock_read_csv:
            mock_df = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
            mock_read_csv.return_value = mock_df
            
            with mock.patch("os.unlink") as mock_unlink:
                # Call the method
                df, error = DataModel.load_dataframe(file)
                
                # Check results
                assert error is None
                assert df is mock_df
                mock_read_csv.assert_called_once_with("tmp_file_path")
                mock_unlink.assert_called_once_with("tmp_file_path")


def test_load_multiple_dataframes_unique_names():
    """Test loading multiple dataframes with unique names."""
    # Create mock files with unique names
    file1 = mock.Mock(spec=UploadedFile)
    file1.name = "customers.csv"
    
    file2 = mock.Mock(spec=UploadedFile)
    file2.name = "orders.csv"
    
    # Mock the load_dataframe method
    with mock.patch.object(DataModel, "load_dataframe") as mock_load:
        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"b": [3, 4]})
        
        # Configure mock to return different dataframes for each file
        mock_load.side_effect = [(df1, None), (df2, None)]
        
        # Call the method
        dataframes, errors = DataModel.load_multiple_dataframes([file1, file2])
        
        # Check results
        assert len(errors) == 0
        assert len(dataframes) == 2
        assert "customers" in dataframes
        assert "orders" in dataframes
        assert dataframes["customers"] is df1
        assert dataframes["orders"] is df2


def test_load_multiple_dataframes_duplicate_names():
    """Test loading multiple dataframes with duplicate names."""
    files = mock_duplicate_files()
    
    # Mock the load_dataframe method
    with mock.patch.object(DataModel, "load_dataframe") as mock_load:
        df1 = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df2 = pd.DataFrame({"col3": [3, 4], "col4": ["c", "d"]})
        df3 = pd.DataFrame({"col5": [5, 6], "col6": ["e", "f"]})
        
        # Configure mock to return different dataframes for each file
        mock_load.side_effect = [(df1, None), (df2, None), (df3, None)]
        
        # Call the method
        dataframes, errors = DataModel.load_multiple_dataframes(files)
        
        # Check results
        assert len(errors) == 0
        assert len(dataframes) == 3
        
        # Check the names have been made unique with version numbers
        assert "data" in dataframes
        assert "data_v1" in dataframes
        assert "data_v2" in dataframes
        
        # Verify the correct dataframes are associated with each name
        # Note: exact name assignment depends on processing order
        assert dataframes["data"] is df1
        assert dataframes["data_v1"] is df2
        assert dataframes["data_v2"] is df3


def test_load_multiple_dataframes_with_errors():
    """Test handling errors when loading multiple dataframes."""
    file1 = mock.Mock(spec=UploadedFile)
    file1.name = "valid.csv"
    
    file2 = mock.Mock(spec=UploadedFile)
    file2.name = "error.csv"
    
    # Mock the load_dataframe method
    with mock.patch.object(DataModel, "load_dataframe") as mock_load:
        df1 = pd.DataFrame({"a": [1, 2]})
        
        # Return a success for first file and an error for second file
        mock_load.side_effect = [(df1, None), (None, "File read error")]
        
        # Call the method
        dataframes, errors = DataModel.load_multiple_dataframes([file1, file2])
        
        # Check results
        assert len(errors) == 1
        assert "error.csv" in errors[0]
        assert len(dataframes) == 1
        assert "valid" in dataframes
        assert dataframes["valid"] is df1 