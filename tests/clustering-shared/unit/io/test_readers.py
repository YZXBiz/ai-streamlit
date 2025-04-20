"""Tests for the IO readers in the shared package."""

import os
import pickle
import tempfile
from io import BytesIO
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


@pytest.fixture
def mock_blob_service():
    """Create a mock BlobServiceClient."""
    mock_service = MagicMock(spec=BlobServiceClient)
    mock_container = MagicMock()
    mock_blob = MagicMock()

    # Set up the chain of mock objects
    mock_service.get_container_client.return_value = mock_container
    mock_container.get_blob_client.return_value = mock_blob

    # Create a mock download object
    mock_download = MagicMock()

    # Default response is CSV data
    mock_download.readall.return_value = b"id,name,value\n1,Alice,10.5\n2,Bob,20.5\n3,Charlie,30.5"
    mock_blob.download_blob.return_value = mock_download

    return {
        "service": mock_service,
        "container": mock_container,
        "blob": mock_blob,
        "download": mock_download,
    }


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
                # Filter data where col1 > 2
                return data.filter(pl.col("col1") > 2)

        # Test the workflow
        reader = TrackedReader(limit=2)  # Set limit to 2 rows
        result = reader.read()

        # Check the call sequence
        assert call_sequence == ["validate", "read", "post_process"]

        # Check the result
        # Original data: [1,2,3,4,5]
        # After limit=2: [1,2]
        # After post_process (filter col1 > 2): []
        assert len(result) == 0  # No rows match the filter after limiting

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

        # Create a concrete test class
        class TestFileReader(FileReader):
            def _read_from_source(self) -> pl.DataFrame:
                return pl.DataFrame()

        with pytest.raises(FileNotFoundError):
            TestFileReader(path="/path/to/nonexistent/file.txt")._validate_source()

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

    @pytest.fixture
    def empty_json_file(self) -> Path:
        """Create an empty JSON file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = Path(temp.name)
            # File is created empty

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

    def test_json_reader_empty_file_lines(self, empty_json_file: Path) -> None:
        """Test JSON reader with empty file and lines=True."""
        reader = JSONReader(path=str(empty_json_file), lines=True)
        result = reader.read()

        # Check result is an empty DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0

    def test_json_reader_empty_file_no_lines(self, empty_json_file: Path) -> None:
        """Test JSON reader with empty file and lines=False."""
        reader = JSONReader(path=str(empty_json_file), lines=False)
        result = reader.read()

        # Check result is an empty DataFrame
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 0


class TestPickleReader:
    """Tests for the PickleReader implementation."""

    @pytest.fixture
    def pickle_file(self) -> Path:
        """Create a temporary pickle file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data and save to pickle
            data = pd.DataFrame(
                {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
            )
            data.to_pickle(temp_path)

        yield temp_path

        # Cleanup
        if temp_path.exists():
            os.unlink(temp_path)

    @pytest.fixture
    def dict_pickle_file(self) -> Path:
        """Create a temporary pickle file with dictionary of DataFrames."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data dictionary and save to pickle
            data_dict = {
                "df1": pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}),
                "df2": pd.DataFrame({"id": [3, 4], "name": ["Charlie", "Dave"]}),
            }
            with open(temp_path, "wb") as f:
                pickle.dump(data_dict, f)

        yield temp_path

    def test_pickle_reader_pandas_df(self, pickle_file: Path) -> None:
        """Test reading pandas DataFrame from pickle."""
        reader = PickleReader(path=str(pickle_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert result["value"].to_list() == [10.5, 20.5, 30.5]

    def test_pickle_reader_polars_df(self, pickle_file: Path) -> None:
        """Test reading polars DataFrame from pickle."""
        # Create a pickle with a polars DataFrame
        data = pl.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
        )
        # Convert to pandas and save since polars doesn't have direct pickle support
        data.to_pandas().to_pickle(pickle_file)

        reader = PickleReader(path=str(pickle_file))
        result = reader.read()

        # Check result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
        assert result["id"].to_list() == [1, 2, 3]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie"]
        assert result["value"].to_list() == [10.5, 20.5, 30.5]

    def test_pickle_reader_dict_of_dataframes(self, dict_pickle_file: Path) -> None:
        """Test reading dictionary of DataFrames from pickle."""
        reader = PickleReader(path=str(dict_pickle_file))
        result = reader.read()

        # Check result is a dictionary
        assert isinstance(result, dict)
        assert "df1" in result
        assert "df2" in result

        # Check each DataFrame in the dictionary
        assert isinstance(result["df1"], pl.DataFrame)
        assert len(result["df1"]) == 2
        assert result["df1"]["id"].to_list() == [1, 2]
        assert result["df1"]["name"].to_list() == ["Alice", "Bob"]

        assert isinstance(result["df2"], pl.DataFrame)
        assert len(result["df2"]) == 2
        assert result["df2"]["id"].to_list() == [3, 4]
        assert result["df2"]["name"].to_list() == ["Charlie", "Dave"]

    def test_pickle_reader_dict_with_limit(self, dict_pickle_file: Path) -> None:
        """Test reading dictionary of DataFrames with limit applied."""
        reader = PickleReader(path=str(dict_pickle_file), limit=1)
        result = reader.read()

        # Check result is a dictionary
        assert isinstance(result, dict)

        # Check limit was applied to each DataFrame
        assert len(result["df1"]) == 1
        assert len(result["df2"]) == 1

        # Check first row of each DataFrame
        assert result["df1"]["id"].to_list() == [1]
        assert result["df1"]["name"].to_list() == ["Alice"]
        assert result["df2"]["id"].to_list() == [3]
        assert result["df2"]["name"].to_list() == ["Charlie"]

    def test_pickle_reader_invalid_data(self) -> None:
        """Test reading invalid data from pickle."""
        # Create a temporary pickle with non-DataFrame data
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
            temp_path = Path(temp.name)

            # Save a simple list
            data = [1, 2, 3]
            with open(temp_path, "wb") as f:
                pickle.dump(data, f)

        try:
            reader = PickleReader(path=str(temp_path))
            result = reader.read()

            # Should convert to DataFrame
            assert isinstance(result, pl.DataFrame)
            assert len(result) == 3

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)


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
        # Patch pd.read_excel to return known data
        mock_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
        )

        with patch("pandas.read_excel", return_value=mock_df):
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
        # Create mock data for different sheets
        mock_df = pd.DataFrame(
            {"product_id": [101, 102, 103], "product_name": ["Apple", "Banana", "Cherry"]}
        )

        with patch("pandas.read_excel", return_value=mock_df):
            # Read from specific sheet
            reader = ExcelReader(path=str(excel_file), sheet_name="Products")
            result = reader.read()

            # Check result
            assert len(result) == 3
            assert "product_id" in result.columns
            assert "product_name" in result.columns


class TestBlobReader:
    """Tests for the BlobReader implementation."""

    def test_validate_source_valid_formats(self):
        """Test validation of supported file formats."""
        for format in ["csv", "parquet", "json", "excel", "pickle"]:
            reader = BlobReader(
                connection_string="test-connection",
                container_name="test-container",
                blob_path="test.csv",
                file_format=format,
            )
            # Should not raise any exception
            reader._validate_source()

    def test_validate_source_invalid_format(self):
        """Test validation fails with unsupported file format."""
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="invalid_format",
        )
        with pytest.raises(ValueError) as excinfo:
            reader._validate_source()

        assert "Unsupported file format" in str(excinfo.value)
        assert "csv, parquet, json, excel, pickle" in str(excinfo.value)

    def test_get_blob_service_client(self):
        """Test creation of blob service client."""
        with patch("azure.storage.blob.BlobServiceClient.from_connection_string") as mock_create:
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            reader = BlobReader(
                connection_string="test-connection",
                container_name="test-container",
                blob_path="test.csv",
                file_format="csv",
            )

            result = reader._get_blob_service_client()

            # Verify client was created with connection string
            mock_create.assert_called_once_with("test-connection")
            assert result == mock_client

    def test_read_from_source_csv(self, mock_blob_service):
        """Test reading CSV data from blob storage."""
        # Set up the mock to return CSV data
        mock_blob_service[
            "download"
        ].readall.return_value = b"id,name,value\n1,Alice,10.5\n2,Bob,20.5"

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        result = reader._read_from_source()

        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name", "value"]

        # Verify proper method calls
        mock_blob_service["service"].get_container_client.assert_called_once_with("test-container")
        mock_blob_service["container"].get_blob_client.assert_called_once_with("test.csv")
        mock_blob_service["blob"].download_blob.assert_called_once()

    def test_read_from_source_parquet(self, mock_blob_service):
        """Test reading Parquet data from blob storage."""
        # Create a simple parquet file in memory
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]})
        parquet_bytes = BytesIO()
        df.to_parquet(parquet_bytes)
        parquet_bytes.seek(0)

        # Set the mock to return parquet data
        mock_blob_service["download"].readall.return_value = parquet_bytes.getvalue()

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.parquet",
            file_format="parquet",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        with patch("polars.read_parquet", return_value=pl.from_pandas(df)) as mock_read:
            result = reader._read_from_source()

            # Verify polars.read_parquet was called
            assert mock_read.call_count == 1
            assert isinstance(result, pl.DataFrame)

    def test_read_from_source_json(self, mock_blob_service):
        """Test reading JSON data from blob storage."""
        # Set the mock to return JSON data
        mock_blob_service[
            "download"
        ].readall.return_value = (
            b'[{"id":1,"name":"Alice","value":10.5},{"id":2,"name":"Bob","value":20.5}]'
        )

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.json",
            file_format="json",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        with patch(
            "polars.read_json",
            return_value=pl.DataFrame(
                {"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]}
            ),
        ) as mock_read:
            result = reader._read_from_source()

            # Verify polars.read_json was called
            assert mock_read.call_count == 1
            assert isinstance(result, pl.DataFrame)

    def test_read_from_source_excel(self, mock_blob_service):
        """Test reading Excel data from blob storage."""
        # Set the mock to return binary data (we'll mock the actual reading)
        mock_blob_service["download"].readall.return_value = b"excel_data"

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.xlsx",
            file_format="excel",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        with patch(
            "polars.read_excel", return_value=pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        ) as mock_read:
            result = reader._read_from_source()

            # Verify polars.read_excel was called
            assert mock_read.call_count == 1
            assert isinstance(result, pl.DataFrame)

    def test_read_from_source_pickle_polars_df(self, mock_blob_service):
        """Test reading pickled Polars DataFrame from blob storage."""
        # Create a pickled Polars DataFrame
        df = pl.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]})
        pickled_data = pickle.dumps(df)

        mock_blob_service["download"].readall.return_value = pickled_data

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.pkl",
            file_format="pickle",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        result = reader._read_from_source()

        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name", "value"]

    def test_read_from_source_pickle_pandas_df(self, mock_blob_service):
        """Test reading pickled Pandas DataFrame from blob storage."""
        # Create a pickled Pandas DataFrame
        df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]})
        pickled_data = pickle.dumps(df)

        mock_blob_service["download"].readall.return_value = pickled_data

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.pkl",
            file_format="pickle",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        result = reader._read_from_source()

        # Verify the result
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns

    def test_read_from_source_pickle_dict(self, mock_blob_service):
        """Test reading pickled dictionary from blob storage."""
        # Create a pickled dictionary
        data_dict = {"id": [1, 2], "name": ["Alice", "Bob"], "value": [10.5, 20.5]}
        pickled_data = pickle.dumps(data_dict)

        mock_blob_service["download"].readall.return_value = pickled_data

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.pkl",
            file_format="pickle",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        result = reader._read_from_source()

        # Verify the result is created from the dictionary
        assert isinstance(result, pl.DataFrame)
        assert "id" in result.columns
        assert "name" in result.columns
        assert "value" in result.columns

    def test_blob_download_error(self, mock_blob_service):
        """Test handling of download errors."""
        # Configure mock to raise an exception
        mock_blob_service["blob"].download_blob.side_effect = Exception("Download failed")

        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv",
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Attempt to read the data - should raise an exception
        with pytest.raises(Exception) as excinfo:
            reader._read_from_source()

        assert "Download failed" in str(excinfo.value)

    def test_max_concurrency_parameter(self, mock_blob_service):
        """Test the max_concurrency parameter is passed to download_blob."""
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv",
            max_concurrency=10,
        )

        # Override the _get_blob_service_client method
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Read the data
        reader._read_from_source()

        # Verify the max_concurrency parameter was passed
        mock_blob_service["blob"].download_blob.assert_called_once_with(max_concurrency=10)

    def test_integration_with_reader_base_class(self, mock_blob_service):
        """Test integration with the base Reader class workflow."""
        # Set up mock data
        mock_blob_service[
            "download"
        ].readall.return_value = b"id,name,value\n1,Alice,10.5\n2,Bob,20.5\n3,Charlie,30.5"

        # Create reader with limit=2 (should only return first 2 rows)
        reader = BlobReader(
            connection_string="test-connection",
            container_name="test-container",
            blob_path="test.csv",
            file_format="csv",
            limit=2,
        )

        # Override the _get_blob_service_client method to use our mock
        reader._get_blob_service_client = lambda: mock_blob_service["service"]

        # Use the read method from the base class
        result = reader.read()

        # Verify limit was applied
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2

        # Verify the specific rows that were kept
        assert result["name"].to_list() == ["Alice", "Bob"]


class TestSnowflakeReader:
    """Tests for the SnowflakeReader implementation."""

    def test_snowflake_reader_mocked(self) -> None:
        """Test SnowflakeReader with mocked Snowflake connector."""
        # Mock DataFrame result from Snowflake
        mock_df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
        )
        mock_pl_df = pl.from_pandas(mock_df)

        # Create mock connection
        mock_conn = MagicMock()

        # Mock both pl.read_database and the _create_connection method
        with (
            patch("polars.read_database", return_value=mock_pl_df) as mock_read_db,
            patch(
                "clustering.shared.io.readers.snowflake_reader.SnowflakeReader._create_connection",
                return_value=mock_conn,
            ),
        ):
            reader = SnowflakeReader(
                query="SELECT * FROM test_table",
                use_cache=False,  # Disable caching for test simplicity
            )

            # Now read the data
            result = reader.read()

            # Check result
            assert isinstance(result, pl.DataFrame)
            assert len(result) == 3
            assert "id" in result.columns
            assert "name" in result.columns
            assert "value" in result.columns

            # Verify the polars.read_database was called with correct parameters
            mock_read_db.assert_called_once_with(
                query="SELECT * FROM test_table", connection=mock_conn
            )
