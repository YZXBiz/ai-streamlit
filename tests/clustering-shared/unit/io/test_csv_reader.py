"""Tests for CSV reader."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import pytest
import polars as pl
import pandas as pd

from clustering.shared.io.readers.csv_reader import CSVReader


class TestCSVReader:
    """Test suite for CSVReader."""

    def setup_method(self):
        """Set up test fixtures for each test method."""
        self.test_csv_content = """id,name,value
1,Alice,100
2,Bob,200
3,Charlie,300
"""
        self.test_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [100, 200, 300]
        })

    def test_read_from_source_basic(self):
        """Test reading CSV file with default settings."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.test_csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create the reader and read the file
            reader = CSVReader(path=tmp_path)
            result = reader._read_from_source()
            
            # Verify the result
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert result.columns == ["id", "name", "value"]
            assert result.row(0) == (1, "Alice", 100)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_read_with_custom_delimiter(self):
        """Test reading CSV file with custom delimiter."""
        # Create a temporary CSV file with different delimiter
        csv_content = """id;name;value
1;Alice;100
2;Bob;200
3;Charlie;300
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create the reader with custom delimiter and read the file
            reader = CSVReader(
                path=tmp_path,
                delimiter=";"
            )
            result = reader._read_from_source()
            
            # Verify the result
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert result.columns == ["id", "name", "value"]
            assert result.row(0) == (1, "Alice", 100)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_read_with_options(self):
        """Test reading CSV file with additional options."""
        # Create a temporary CSV file with comments and some extra options
        csv_content = """# This is a comment
id,name,value
# Another comment
1,Alice,100
N/A,Bob,200
3,Charlie,NULL
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create the reader with custom options and read the file
            reader = CSVReader(
                path=tmp_path,
                comment_char="#",
                null_values=["N/A", "NULL"]
            )
            result = reader._read_from_source()
            
            # Verify the result
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)  # 3 rows because we're skipping comments
            assert result.columns == ["id", "name", "value"]
            # Check for null values
            assert result["id"][1] is None or pd.isna(result["id"][1])
            assert result["value"][2] is None or pd.isna(result["value"][2])
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_limit_rows(self):
        """Test limiting number of rows."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.test_csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create the reader with limit and read the file
            reader = CSVReader(path=tmp_path, limit=2)
            result = reader.read()  # Use read() instead of _read_from_source()
            
            # Verify the result has the correct number of rows
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (2, 3)
            assert result.row(0) == (1, "Alice", 100)
            assert result.row(1) == (2, "Bob", 200)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_column_selection(self):
        """Test selecting specific columns."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.test_csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create the reader with specific columns
            reader = CSVReader(path=tmp_path, columns=["id", "value"])
            result = reader._read_from_source()
            
            # Verify only requested columns are included
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 2)
            assert set(result.columns) == {"id", "value"}
            assert "name" not in result.columns
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_skip_rows(self):
        """Test skipping rows."""
        # Create a temporary CSV file with header row
        csv_content = """header1,header2,header3
skip1,skip2,skip3
id,name,value
1,Alice,100
2,Bob,200
3,Charlie,300
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create the reader that skips the first two rows
            reader = CSVReader(path=tmp_path, skip_rows=2)
            result = reader._read_from_source()
            
            # Verify the correct rows were read and the headers are correct
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert result.columns == ["id", "name", "value"]
            assert result.row(0) == (1, "Alice", 100)
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_integration_with_reader_base_class(self):
        """Test integration with the Reader base class."""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(self.test_csv_content)
            tmp_path = tmp_file.name
        
        try:
            # Create the reader
            reader = CSVReader(path=tmp_path)
            
            # Call the read method from the base class
            result = reader.read()
            
            # Verify the result
            assert isinstance(result, pl.DataFrame)
            assert result.shape == (3, 3)
            assert result.columns == ["id", "name", "value"]
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path) 