"""Shared fixtures for IO tests."""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "value": [10.1, 20.2, 30.3, 40.4, 50.5],
            "active": [True, False, True, True, False],
        }
    )


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def csv_file(temp_dir, sample_data):
    """Create a temporary CSV file with sample data."""
    file_path = temp_dir / "test_data.csv"
    sample_data.write_csv(file_path)
    yield file_path
    if file_path.exists():
        os.remove(file_path)


@pytest.fixture
def parquet_file(temp_dir, sample_data):
    """Create a temporary Parquet file with sample data."""
    file_path = temp_dir / "test_data.parquet"
    sample_data.write_parquet(file_path)
    yield file_path
    if file_path.exists():
        os.remove(file_path)


@pytest.fixture
def excel_file(temp_dir, sample_data):
    """Create a temporary Excel file with sample data."""
    file_path = temp_dir / "test_data.xlsx"
    sample_data.write_excel(file_path)
    yield file_path
    if file_path.exists():
        os.remove(file_path)


@pytest.fixture
def pickle_file(temp_dir, sample_data):
    """Create a temporary Pickle file with sample data."""
    import pickle

    file_path = temp_dir / "test_data.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(sample_data, f)
    yield file_path
    if file_path.exists():
        os.remove(file_path)
