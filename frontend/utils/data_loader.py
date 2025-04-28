"""
Data loading utilities for the PandasAI Streamlit application.

This module provides functions for loading data from various sources.
"""

import os

import pandas as pd
import pandasai as pai
import streamlit as st

from backend.app.adapters.file_sources import CSVSource, ExcelSource, ParquetSource


def load_file(
    file_path: str, file_type: str, name: str, description: str, **kwargs
) -> pai.DataFrame:
    """
    Load a file into a PandasAI DataFrame.

    Args:
        file_path: Path to the file
        file_type: Type of file ('csv', 'excel', 'parquet')
        name: Name for the dataset
        description: Description of the dataset
        **kwargs: Additional arguments for specific file types

    Returns:
        A PandasAI DataFrame

    Raises:
        ValueError: If the file type is not supported
    """
    if file_type.lower() == "csv":
        source = CSVSource(file_path, name, description)
    elif file_type.lower() in ["xlsx", "xls", "excel"]:
        sheet_name = kwargs.get("sheet_name", None)
        source = ExcelSource(file_path, name, description, sheet_name=sheet_name)
    elif file_type.lower() == "parquet":
        source = ParquetSource(file_path, name, description)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return source.load()


def create_sample_data() -> pai.DataFrame:
    """
    Create a sample dataset for demonstration purposes.

    Returns:
        A PandasAI DataFrame with sample data
    """
    # Create sample data
    sample_data = pd.DataFrame(
        {
            "country": [
                "United States",
                "United Kingdom",
                "France",
                "Germany",
                "Italy",
                "Spain",
                "Canada",
                "Australia",
                "Japan",
                "China",
            ],
            "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000],
            "employees": [150, 90, 80, 120, 70, 65, 85, 80, 130, 200],
            "year_founded": [1980, 1985, 1990, 1975, 1995, 2000, 1988, 1992, 1970, 1965],
        }
    )

    # Convert to PandasAI DataFrame with v3 syntax
    return pai.DataFrame(
        sample_data,
        name="sample_data",
        description="Sample company data with revenue, employees, and founding year",
    )
