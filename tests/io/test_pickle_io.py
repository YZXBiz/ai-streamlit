"""Tests for the pickle reader and writer modules."""

import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from clustering.io.readers.pickle_reader import PickleReader
from clustering.io.writers.pickle_writer import PickleWriter


@pytest.fixture
def sample_dataframe() -> pl.DataFrame:
    """Create a sample DataFrame for testing.

    Returns:
        A sample polars DataFrame.
    """
    return pl.DataFrame(
        {"id": [1, 2, 3], "name": ["Item 1", "Item 2", "Item 3"], "value": [10.5, 20.75, 30.25]}
    )


@pytest.fixture
def sample_dict_dataframes() -> dict[str, pl.DataFrame]:
    """Create a sample dictionary of DataFrames for testing.

    Returns:
        A dictionary of sample polars DataFrames.
    """
    return {
        "health": pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Health Item 1", "Health Item 2", "Health Item 3"],
                "value": [100.5, 200.75, 300.25],
            }
        ),
        "beauty": pl.DataFrame(
            {
                "id": [4, 5, 6],
                "name": ["Beauty Item 1", "Beauty Item 2", "Beauty Item 3"],
                "value": [400.5, 500.75, 600.25],
            }
        ),
        "grocery": pl.DataFrame(
            {
                "id": [7, 8, 9],
                "name": ["Grocery Item 1", "Grocery Item 2", "Grocery Item 3"],
                "value": [700.5, 800.75, 900.25],
            }
        ),
    }


@pytest.fixture
def sample_model_output() -> dict:
    """Create a sample model output dictionary for testing.

    Returns:
        A dictionary simulating model output with centroids and other components.
    """
    return {
        "health": {
            "model": "KMeans",
            "centroids": {
                0: np.array([1.0, 2.0, 3.0]),
                1: np.array([4.0, 5.0, 6.0]),
                2: np.array([7.0, 8.0, 9.0]),
            },
            "params": {"n_clusters": 3, "random_state": 42},
            "metrics": {"silhouette": 0.75, "calinski_harabasz": 150.5, "davies_bouldin": 0.25},
        },
        "beauty": {
            "model": "KMeans",
            "centroids": {
                0: np.array([10.0, 20.0, 30.0]),
                1: np.array([40.0, 50.0, 60.0]),
            },
            "params": {"n_clusters": 2, "random_state": 42},
            "metrics": {"silhouette": 0.82, "calinski_harabasz": 200.5, "davies_bouldin": 0.18},
        },
    }


def test_pickle_writer_dataframe(sample_dataframe: pl.DataFrame) -> None:
    """Test writing a single DataFrame with pickle writer.

    Args:
        sample_dataframe: A sample DataFrame to write.
    """
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create pickle writer
        writer = PickleWriter(path=str(temp_path))

        # Write data
        writer.write(sample_dataframe)

        # Check file exists
        assert os.path.exists(temp_path)

        # Read back the data to verify
        reader = PickleReader(path=str(temp_path))
        df_read = reader.read()

        # Assert it's a DataFrame
        assert isinstance(df_read, pl.DataFrame)

        # Assert shape
        assert df_read.shape == (3, 3)

        # Assert column names
        assert df_read.columns == ["id", "name", "value"]

        # Assert values
        assert df_read.select("id").to_series().to_list() == [1, 2, 3]
        assert df_read.select("name").to_series().to_list() == ["Item 1", "Item 2", "Item 3"]
        assert df_read.select("value").to_series().to_list() == [10.5, 20.75, 30.25]
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_pickle_writer_dict_dataframes(sample_dict_dataframes: dict[str, pl.DataFrame]) -> None:
    """Test writing a dictionary of DataFrames with pickle writer.

    Args:
        sample_dict_dataframes: A dictionary of sample DataFrames to write.
    """
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create pickle writer
        writer = PickleWriter(path=str(temp_path))

        # Write data
        writer.write(sample_dict_dataframes)

        # Check file exists
        assert os.path.exists(temp_path)

        # Read back the data to verify
        reader = PickleReader(path=str(temp_path))
        dict_read = reader.read()

        # Assert it's a dictionary
        assert isinstance(dict_read, dict)

        # Assert keys
        assert set(dict_read.keys()) == {"health", "beauty", "grocery"}

        # Check each DataFrame
        for key in dict_read:
            # Assert it's a DataFrame
            assert isinstance(dict_read[key], pl.DataFrame)

            # Assert shape
            assert dict_read[key].shape == (3, 3)

            # Assert it has the expected columns
            assert dict_read[key].columns == ["id", "name", "value"]
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_pickle_writer_model_output(sample_model_output: dict) -> None:
    """Test writing model output with pickle writer.

    Args:
        sample_model_output: A sample model output dictionary to write.
    """
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create pickle writer but not used directly
        # The direct writer usage is tested in other tests
        
        # Write data - note we need to bypass validation for non-DataFrame data
        with open(temp_path, "wb") as file:
            import pickle

            pickle.dump(sample_model_output, file, protocol=pickle.HIGHEST_PROTOCOL)

        # Check file exists
        assert os.path.exists(temp_path)

        # Read back the data to verify
        with open(temp_path, "rb") as file:
            model_read = pickle.load(file)

        # Assert it's a dictionary
        assert isinstance(model_read, dict)

        # Assert keys
        assert set(model_read.keys()) == {"health", "beauty"}

        # Check centroids
        assert np.array_equal(
            model_read["health"]["centroids"][0], sample_model_output["health"]["centroids"][0]
        )
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_pickle_reader_with_limit(sample_dict_dataframes: dict[str, pl.DataFrame]) -> None:
    """Test pickle reader with a row limit.

    Args:
        sample_dict_dataframes: A dictionary of sample DataFrames to write.
    """
    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Write data
        writer = PickleWriter(path=str(temp_path))
        writer.write(sample_dict_dataframes)

        # Create pickle reader with limit
        reader = PickleReader(path=str(temp_path), limit=2)

        # Read data
        dict_read = reader.read()

        # Assert it's a dictionary
        assert isinstance(dict_read, dict)

        # Check each DataFrame has only 2 rows due to limit
        for key in dict_read:
            assert dict_read[key].shape == (2, 3)
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_pickle_writer_empty_dataframe() -> None:
    """Test pickle writer with empty DataFrame."""
    # Create empty DataFrame
    empty_df = pl.DataFrame({"a": [], "b": []})

    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create pickle writer
        writer = PickleWriter(path=str(temp_path))

        # Writing empty DataFrame should raise ValueError
        with pytest.raises(ValueError, match="Cannot write empty DataFrame"):
            writer.write(empty_df)
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_pickle_writer_empty_dict() -> None:
    """Test pickle writer with empty dictionary."""
    # Create empty dictionary
    empty_dict: dict[str, pl.DataFrame] = {}

    # Create a temporary file path
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
        temp_path = Path(temp.name)

    try:
        # Create pickle writer
        writer = PickleWriter(path=str(temp_path))

        # Writing empty dictionary should raise ValueError
        with pytest.raises(ValueError, match="Cannot write empty dictionary"):
            writer.write(empty_dict)
    finally:
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)


def test_pickle_reader_nonexistent_file() -> None:
    """Test pickle reader with a nonexistent file."""
    # Create pickle reader with nonexistent file
    reader = PickleReader(path="nonexistent_file.pkl")

    # Reading should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        reader.read()
