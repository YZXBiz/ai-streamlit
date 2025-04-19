"""Tests for the IO readers in the shared package."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
from azure.storage.blob import BlobServiceClient

from clustering.shared.io.readers import (
    BlobReader,
    CSVReader,
    ExcelReader,
    FileReader,
    JSONReader,
    ParquetReader,
    PickleReader,
    Reader,
    SnowflakeReader,
)


class TestBaseReader:
    """Tests for the base Reader class."""

    def test_reader_abstract_methods(self) -> None:
        """Test that Reader abstract methods must be implemented."""
        # Attempt to instantiate abstract class
        with pytest.raises(TypeError):
            Reader()  # type: ignore

        # Create a subclass without implementing abstract methods
        class IncompleteReader(Reader):
            pass

        with pytest.raises(TypeError):
            IncompleteReader()  # type: ignore

        # Create a minimal valid implementation
        class MinimalReader(Reader):
            def _read_from_source(self) -> pl.DataFrame:
                return pl.DataFrame({"col1": [1, 2, 3]})

        # This should work
        reader = MinimalReader()
        result = reader.read()
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_reader_template_method(self) -> None:
        """Test the template method design pattern in Reader."""
        # Create a spy to track method calls
        call_sequence = []

        class TrackedReader(Reader):
            def _validate_source(self) -> None:
                call_sequence.append("validate")
                super()._validate_source()

            def _read_from_source(self) -> pl.DataFrame:
                call_sequence.append("read")
                return pl.DataFrame({"col1": [1, 2, 3, 4, 5]})

            def _post_process(self, data: pl.DataFrame) -> pl.DataFrame:
                call_sequence.append("post_process")
                return data.filter(pl.col("col1") > 2)  # Filter out some rows

        # Test the workflow
        reader = TrackedReader(limit=2)  # Set limit to 2 rows
        result = reader.read()

        # Check the call sequence
        assert call_sequence == ["validate", "read", "post_process"]

        # Check the result
        assert len(result) == 2  # Limit applied after filtering
        assert result["col1"].to_list() == [3, 4]  # First two rows after filtering

    def test_reader_limit(self) -> None:
        """Test that limit parameter works correctly."""

        class LimitTestReader(Reader):
            def _read_from_source(self) -> pl.DataFrame:
                return pl.DataFrame({"col1": list(range(100))})

        # Test without limit
        reader = LimitTestReader()
        result = reader.read()
        assert len(result) == 100

        # Test with limit
        reader = LimitTestReader(limit=10)
        result = reader.read()
        assert len(result) == 10
        assert result["col1"].to_list() == list(range(10))


class TestFileReader:
    """Tests for the FileReader base class."""

    def test_file_reader_validation(self) -> None:
        """Test that FileReader validates file existence."""
        with pytest.raises(FileNotFoundError):
            FileReader(path="/path/to/nonexistent/file.txt")._validate_source()

    def test_file_reader_str_representation(self) -> None:
        """Test string representation of FileReader."""

        # Create a concrete subclass for testing
        class ConcreteFileReader(FileReader):
            def _read_from_source(self) -> pl.DataFrame:
                return pl.DataFrame()

        reader = ConcreteFileReader(path="/path/to/file.txt")
        assert "ConcreteFileReader" in str(reader)
        assert "/path/to/file.txt" in str(reader)


class TestCSVReader:
    """Tests for the CSVReader implementation."""

    @pytest.fixture
    def csv_file(self) -> Path:
        """Create a temporary CSV file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data
            data = pd.DataFrame(
                {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
            )

            # Write data to CSV
            data.to_csv(temp_path, index=False)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)

    def test_csv_reader_basic(self, csv_file: Path) -> None:
        """Test basic CSV reading functionality."""
        reader = CSVReader(path=str(csv_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie"]

    def test_csv_reader_options(self, csv_file: Path) -> None:
        """Test CSV reader with various options."""
        # Test with different delimiter and column selection
        with open(csv_file, "w") as f:
            f.write("id|name|value\n")
            f.write("1|Alice|10.5\n")
            f.write("2|Bob|20.5\n")
            f.write("3|Charlie|30.5\n")

        reader = CSVReader(
            path=str(csv_file),
            delimiter="|",
            columns=["id", "name"],  # Only read these columns
        )
        result = reader.read()

        # Check result
        assert len(result) == 3
        assert result.columns == ["id", "name"]
        assert "value" not in result.columns

    def test_csv_reader_limit(self, csv_file: Path) -> None:
        """Test CSV reader with row limit."""
        reader = CSVReader(path=str(csv_file), limit=2)
        result = reader.read()

        # Check result
        assert len(result) == 2
        assert result["id"].to_list() == [1, 2]


class TestParquetReader:
    """Tests for the ParquetReader implementation."""

    @pytest.fixture
    def parquet_file(self) -> Path:
        """Create a temporary Parquet file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data
            data = pd.DataFrame(
                {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
            )

            # Write data to Parquet
            data.to_parquet(temp_path, index=False)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)

    def test_parquet_reader(self, parquet_file: Path) -> None:
        """Test basic Parquet reading functionality."""
        reader = ParquetReader(path=str(parquet_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie"]


class TestJSONReader:
    """Tests for the JSONReader implementation."""

    @pytest.fixture
    def json_file(self) -> Path:
        """Create a temporary JSON file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data
            data = pd.DataFrame(
                {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
            )

            # Write data to JSON
            data.to_json(temp_path, orient="records", lines=True)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)

    def test_json_reader(self, json_file: Path) -> None:
        """Test basic JSON reading functionality."""
        reader = JSONReader(path=str(json_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns

    def test_json_reader_options(self, json_file: Path) -> None:
        """Test JSON reader with various options."""
        # Write a non-lines JSON file
        with open(json_file, "w") as f:
            f.write("""
            [
                {"id": 1, "name": "Alice", "value": 10.5},
                {"id": 2, "name": "Bob", "value": 20.5},
                {"id": 3, "name": "Charlie", "value": 30.5}
            ]
            """)

        reader = JSONReader(path=str(json_file), lines=False)
        result = reader.read()

        # Check result
        assert len(result) == 3
        assert result["id"].to_list() == [1, 2, 3]


class TestPickleReader:
    """Tests for the PickleReader implementation."""

    @pytest.fixture
    def pickle_file(self) -> Path:
        """Create a temporary pickle file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data
            data = pd.DataFrame(
                {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
            )

            # Write data to pickle
            data.to_pickle(temp_path)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)

    def test_pickle_reader_pandas_df(self, pickle_file: Path) -> None:
        """Test reading pandas DataFrame from pickle."""
        reader = PickleReader(path=str(pickle_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns

    def test_pickle_reader_polars_df(self, pickle_file: Path) -> None:
        """Test reading polars DataFrame from pickle."""
        # Create a pickle with a polars DataFrame
        data = pl.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
        )
        data.write_pickle(pickle_file)

        reader = PickleReader(path=str(pickle_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result["id"].to_list() == [1, 2, 3]


class TestExcelReader:
    """Tests for the ExcelReader implementation."""

    @pytest.fixture
    def excel_file(self) -> Path:
        """Create a temporary Excel file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data
            data = pd.DataFrame(
                {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
            )

            # Write data to Excel
            data.to_excel(temp_path, index=False)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)

    def test_excel_reader(self, excel_file: Path) -> None:
        """Test basic Excel reading functionality."""
        reader = ExcelReader(path=str(excel_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns

    def test_excel_reader_sheet_name(self, excel_file: Path) -> None:
        """Test Excel reader with specific sheet name."""
        # Create Excel with multiple sheets
        with pd.ExcelWriter(excel_file) as writer:
            pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}).to_excel(
                writer, sheet_name="Sheet1", index=False
            )

            pd.DataFrame(
                {"product_id": [101, 102, 103], "product_name": ["Apple", "Banana", "Cherry"]}
            ).to_excel(writer, sheet_name="Products", index=False)

        # Read from specific sheet
        reader = ExcelReader(path=str(excel_file), sheet_name="Products")
        result = reader.read()

        # Check result
        assert len(result) == 3
        assert "product_id" in result.columns
        assert "product_name" in result.columns


class TestBlobReader:
    """Tests for the BlobReader implementation."""

    def test_blob_reader_mocked(self) -> None:
        """Test BlobReader with mocked Azure client."""
        # Create mock objects
        mock_blob_service = MagicMock(spec=BlobServiceClient)
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        # Set up the chain of mock objects
        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client

        # Set up download_blob to return CSV data
        mock_download = MagicMock()
        mock_download.readall.return_value = (
            b"id,name,value\n1,Alice,10.5\n2,Bob,20.5\n3,Charlie,30.5"
        )
        mock_blob_client.download_blob.return_value = mock_download

        # Create reader with patch
        with patch(
            "clustering.shared.io.readers.blob_reader.BlobServiceClient",
            return_value=mock_blob_service,
        ):
            reader = BlobReader(
                connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net",
                container_name="test-container",
                blob_path="data/test.csv",
                file_format="csv",
            )
            result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns

        # Verify proper sequence of calls
        mock_blob_service.get_container_client.assert_called_once_with("test-container")
        mock_container_client.get_blob_client.assert_called_once_with("data/test.csv")
        mock_blob_client.download_blob.assert_called_once()

    @pytest.mark.parametrize("file_format", ["csv", "parquet", "json", "excel", "pickle"])
    def test_blob_reader_format_validation(self, file_format: str) -> None:
        """Test that BlobReader validates file format."""
        # These should not raise errors
        BlobReader(
            connection_string="test",
            container_name="test",
            blob_path="test.csv",
            file_format=file_format,
        )

        # Invalid format should raise error
        with pytest.raises(ValueError):
            BlobReader(
                connection_string="test",
                container_name="test",
                blob_path="test.csv",
                file_format="invalid_format",  # Invalid format
            )


class TestSnowflakeReader:
    """Tests for the SnowflakeReader implementation."""

    def test_snowflake_reader_mocked(self) -> None:
        """Test SnowflakeReader with mocked Snowflake connector."""
        # Mock DataFrame result from Snowflake
        mock_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
        )

        # Mock Snowflake connector
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_cursor.fetch_pandas_all.return_value = mock_df

        # Patch snowflake.connector.connect
        with patch(
            "clustering.shared.io.readers.snowflake_reader.snowflake.connector.connect",
            return_value=mock_conn,
        ):
            reader = SnowflakeReader(
                user="test_user",
                password="test_password",
                account="test_account",
                database="test_db",
                schema="test_schema",
                warehouse="test_warehouse",
                query="SELECT * FROM test_table",
            )
            result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns

        # Verify query execution
        mock_cursor.execute.assert_called_once_with("SELECT * FROM test_table")
