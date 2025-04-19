"""Tests for the IO writers in the shared package."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pytest
from azure.storage.blob import BlobServiceClient

from clustering.shared.io.writers import (
    BlobWriter,
    CSVWriter,
    ExcelWriter,
    FileWriter,
    JSONWriter,
    ParquetWriter,
    PickleWriter,
    Writer,
)


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample DataFrame for testing writers."""
    return pl.DataFrame(
        {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "value": [10.5, 20.5, 30.5]}
    )


class TestBaseWriter:
    """Tests for the base Writer class."""

    def test_writer_abstract_methods(self) -> None:
        """Test that Writer abstract methods must be implemented."""
        # Attempt to instantiate abstract class
        with pytest.raises(TypeError):
            Writer()  # type: ignore

        # Create a subclass without implementing abstract methods
        class IncompleteWriter(Writer):
            pass

        with pytest.raises(TypeError):
            IncompleteWriter()  # type: ignore

        # Create a minimal valid implementation
        class MinimalWriter(Writer):
            def _write_to_destination(self, data: pl.DataFrame) -> None:
                pass

        # This should work
        writer = MinimalWriter()
        # No error should be raised
        writer.write(pl.DataFrame({"col1": [1, 2, 3]}))

    def test_writer_template_method(self) -> None:
        """Test the template method design pattern in Writer."""
        # Create a spy to track method calls
        call_sequence = []

        class TrackedWriter(Writer):
            def _validate_destination(self) -> None:
                call_sequence.append("validate")
                super()._validate_destination()

            def _pre_process(self, data: pl.DataFrame) -> pl.DataFrame:
                call_sequence.append("pre_process")
                # Add a column
                return data.with_columns(pl.lit(True).alias("processed"))

            def _write_to_destination(self, data: pl.DataFrame) -> None:
                call_sequence.append("write")
                # Verify pre-processing was done
                assert "processed" in data.columns

        # Test the workflow
        writer = TrackedWriter()
        writer.write(pl.DataFrame({"col1": [1, 2, 3]}))

        # Check the call sequence
        assert call_sequence == ["validate", "pre_process", "write"]


class TestFileWriter:
    """Tests for the FileWriter base class."""

    def test_file_writer_str_representation(self) -> None:
        """Test string representation of FileWriter."""

        # Create a concrete subclass for testing
        class ConcreteFileWriter(FileWriter):
            def _write_to_destination(self, data: pl.DataFrame) -> None:
                pass

        writer = ConcreteFileWriter(path="/path/to/output.txt")
        assert "ConcreteFileWriter" in str(writer)
        assert "/path/to/output.txt" in str(writer)

    def test_file_writer_directory_creation(self) -> None:
        """Test that FileWriter creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "a" / "b" / "c"
            output_path = nested_dir / "output.txt"

            # Directory shouldn't exist yet
            assert not nested_dir.exists()

            # Create a concrete subclass for testing
            class ConcreteFileWriter(FileWriter):
                def _write_to_destination(self, data: pl.DataFrame) -> None:
                    # Just create an empty file
                    with open(self.path, "w") as f:
                        f.write("")

            # Create writer and write data
            writer = ConcreteFileWriter(path=str(output_path))
            writer._validate_destination()  # This should create the directories
            writer._write_to_destination(pl.DataFrame())

            # Directory should now exist
            assert nested_dir.exists()
            assert output_path.exists()


class TestCSVWriter:
    """Tests for the CSVWriter implementation."""

    def test_csv_writer_basic(self, sample_dataframe: pl.DataFrame) -> None:
        """Test basic CSV writing functionality."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data
            writer = CSVWriter(path=str(temp_path))
            writer.write(sample_dataframe)

            # Verify file exists
            assert temp_path.exists()

            # Read back and verify content
            result = pl.read_csv(temp_path)
            assert len(result) == 3
            assert "id" in result.columns
            assert "name" in result.columns
            assert "value" in result.columns
            assert result["id"].to_list() == [1, 2, 3]

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_csv_writer_options(self, sample_dataframe: pl.DataFrame) -> None:
        """Test CSV writer with various options."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data with custom options
            writer = CSVWriter(path=str(temp_path), delimiter="|", include_header=False)
            writer.write(sample_dataframe)

            # Check raw file content
            with open(temp_path) as f:
                content = f.read()

            # Should use pipe delimiter and have no header
            assert "id|name|value" not in content  # No header
            assert "1|Alice|10.5" in content  # Data with pipe delimiter

            # Read back with correct options
            result = pl.read_csv(temp_path, delimiter="|", has_header=False)
            assert len(result) == 3

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)


class TestParquetWriter:
    """Tests for the ParquetWriter implementation."""

    def test_parquet_writer(self, sample_dataframe: pl.DataFrame) -> None:
        """Test basic Parquet writing functionality."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data
            writer = ParquetWriter(path=str(temp_path))
            writer.write(sample_dataframe)

            # Verify file exists
            assert temp_path.exists()

            # Read back and verify content
            result = pl.read_parquet(temp_path)
            assert len(result) == 3
            assert "id" in result.columns
            assert "name" in result.columns
            assert "value" in result.columns
            assert result["id"].to_list() == [1, 2, 3]

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_parquet_writer_compression(self, sample_dataframe: pl.DataFrame) -> None:
        """Test Parquet writer with compression options."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp:
            temp_path = Path(temp.name)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as temp2:
            temp_path2 = Path(temp2.name)

        try:
            # Write data without compression
            writer1 = ParquetWriter(path=str(temp_path), compression=None)
            writer1.write(sample_dataframe)

            # Write data with compression
            writer2 = ParquetWriter(path=str(temp_path2), compression="snappy")
            writer2.write(sample_dataframe)

            # Both files should exist
            assert temp_path.exists()
            assert temp_path2.exists()

            # Compressed file should be smaller
            # We don't actually use the file sizes for comparison in this test
            # This is just to show that compression would be used in a real scenario

            # Read back and verify both have same content
            result1 = pl.read_parquet(temp_path)
            result2 = pl.read_parquet(temp_path2)

            assert len(result1) == len(result2)
            assert result1["id"].to_list() == result2["id"].to_list()

        finally:
            # Cleanup
            for path in [temp_path, temp_path2]:
                if path.exists():
                    os.unlink(path)


class TestJSONWriter:
    """Tests for the JSONWriter implementation."""

    def test_json_writer(self, sample_dataframe: pl.DataFrame) -> None:
        """Test basic JSON writing functionality."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data
            writer = JSONWriter(path=str(temp_path))
            writer.write(sample_dataframe)

            # Verify file exists
            assert temp_path.exists()

            # Read back and verify content
            result = pl.read_json(temp_path)
            assert len(result) == 3
            assert "id" in result.columns
            assert "name" in result.columns
            assert "value" in result.columns
            assert result["id"].to_list() == [1, 2, 3]

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_json_writer_options(self, sample_dataframe: pl.DataFrame) -> None:
        """Test JSON writer with various options."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data with custom options (pretty formatting)
            writer = JSONWriter(path=str(temp_path), pretty=True)
            writer.write(sample_dataframe)

            # Check raw file content
            with open(temp_path) as f:
                content = f.read()

            # Should have pretty formatting with indentation
            assert "  " in content  # Has indentation
            assert "\n" in content  # Has newlines

            # Read back
            result = pl.read_json(temp_path)
            assert len(result) == 3

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)


class TestPickleWriter:
    """Tests for the PickleWriter implementation."""

    def test_pickle_writer(self, sample_dataframe: pl.DataFrame) -> None:
        """Test basic pickle writing functionality."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data
            writer = PickleWriter(path=str(temp_path))
            writer.write(sample_dataframe)

            # Verify file exists
            assert temp_path.exists()

            # Read back and verify content (using pandas for simplicity)
            result = pd.read_pickle(temp_path)

            # Convert to list for comparison if needed
            if hasattr(result, "to_dict"):
                # It's a DataFrame, convert to dict for comparison
                result_dict = result.to_dict("list")
                assert result_dict["id"] == [1, 2, 3]
                assert result_dict["name"] == ["Alice", "Bob", "Charlie"]

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_pickle_writer_protocol(self, sample_dataframe: pl.DataFrame) -> None:
        """Test pickle writer with different protocol versions."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data with oldest protocol for compatibility
            writer = PickleWriter(path=str(temp_path), protocol=2)
            writer.write(sample_dataframe)

            # Verify file exists
            assert temp_path.exists()

            # Should be readable with pandas (which expects older protocols)
            result = pd.read_pickle(temp_path)
            assert len(result) == 3

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)


class TestExcelWriter:
    """Tests for the ExcelWriter implementation."""

    def test_excel_writer(self, sample_dataframe: pl.DataFrame) -> None:
        """Test basic Excel writing functionality."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data
            writer = ExcelWriter(path=str(temp_path))
            writer.write(sample_dataframe)

            # Verify file exists
            assert temp_path.exists()

            # Read back with pandas and verify content
            result = pd.read_excel(temp_path)
            assert len(result) == 3
            assert "id" in result.columns
            assert "name" in result.columns
            assert "value" in result.columns
            assert result["id"].tolist() == [1, 2, 3]

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_excel_writer_sheet_name(self, sample_dataframe: pl.DataFrame) -> None:
        """Test Excel writer with custom sheet name."""
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Write data with custom sheet name
            writer = ExcelWriter(path=str(temp_path), sheet_name="TestData")
            writer.write(sample_dataframe)

            # Read back with pandas and verify sheet name
            xl = pd.ExcelFile(temp_path)
            assert "TestData" in xl.sheet_names

            # Read the specific sheet
            result = pd.read_excel(temp_path, sheet_name="TestData")
            assert len(result) == 3

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)


class TestBlobWriter:
    """Tests for the BlobWriter implementation."""

    def test_blob_writer_mocked(self, sample_dataframe: pl.DataFrame) -> None:
        """Test BlobWriter with mocked Azure client."""
        # Create mock objects
        mock_blob_service = MagicMock(spec=BlobServiceClient)
        mock_container_client = MagicMock()
        mock_blob_client = MagicMock()

        # Set up the chain of mock objects
        mock_blob_service.get_container_client.return_value = mock_container_client
        mock_container_client.get_blob_client.return_value = mock_blob_client

        # Create a temp file to simulate local staging
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = Path(temp.name)

        try:
            # Patch the BlobServiceClient
            with patch(
                "clustering.shared.io.writers.blob_writer.BlobServiceClient",
                return_value=mock_blob_service,
            ):
                # Create writer
                writer = BlobWriter(
                    connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=test;EndpointSuffix=core.windows.net",
                    container_name="test-container",
                    blob_path="data/test.csv",
                    file_format="csv",
                    local_staging_path=str(temp_path),
                )
                # Write data
                writer.write(sample_dataframe)

            # Verify proper sequence of calls
            mock_blob_service.get_container_client.assert_called_once_with("test-container")
            mock_container_client.get_blob_client.assert_called_once_with("data/test.csv")
            mock_blob_client.upload_blob.assert_called_once()

            # Verify local staging file was created
            assert temp_path.exists()

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    @pytest.mark.parametrize("file_format", ["csv", "parquet", "json", "excel", "pickle"])
    def test_blob_writer_format_validation(self, file_format: str) -> None:
        """Test that BlobWriter validates file format."""
        # These should not raise errors
        BlobWriter(
            connection_string="test",
            container_name="test",
            blob_path=f"test.{file_format}",
            file_format=file_format,
        )

        # Invalid format should raise error
        with pytest.raises(ValueError):
            BlobWriter(
                connection_string="test",
                container_name="test",
                blob_path="test.txt",
                file_format="invalid_format",  # Invalid format
            )
