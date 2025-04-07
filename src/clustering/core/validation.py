"""Validation utilities for the clustering pipeline.

This module provides utilities for validating data structures
before operations to catch errors early with helpful messages.
"""

import polars as pl


def validate_columns(df: pl.DataFrame, required_columns: list[str], context: str = "") -> list[str]:
    """Validate that a DataFrame contains the required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        context: Optional context description for better error messages

    Returns:
        List of missing columns (empty if all required columns are present)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return missing_columns


def assert_columns(df: pl.DataFrame, required_columns: list[str], context: str = "") -> None:
    """Assert that a DataFrame contains the required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        context: Optional context description for better error messages

    Raises:
        ValueError: If any required columns are missing
    """
    missing_columns = validate_columns(df, required_columns, context)
    if missing_columns:
        context_msg = f" in {context}" if context else ""
        raise ValueError(
            f"Missing required columns{context_msg}: {', '.join(missing_columns)}.\n"
            f"Available columns: {', '.join(df.columns)}"
        )
