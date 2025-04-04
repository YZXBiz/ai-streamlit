"""Validation utilities for the clustering pipeline.

This module provides utilities for validating data structures
before operations to catch errors early with helpful messages.
"""

from typing import Any

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


def check_column_types(
    df: pl.DataFrame, expected_types: dict[str, type | list[type]], context: str = ""
) -> dict[str, tuple[Any, str]]:
    """Check that columns have the expected types.

    Args:
        df: DataFrame to validate
        expected_types: Dict mapping column names to expected types
        context: Optional context description for better error messages

    Returns:
        Dict mapping column names to tuples of (expected_type, actual_type) for mismatches
    """
    mismatches = {}
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            continue

        # Get the Polars dtype object
        actual_dtype = df[col].dtype

        # Convert to Python type name for comparison
        actual_type_name = str(actual_dtype)

        # Handle case where multiple types are acceptable
        if isinstance(expected_type, list):
            expected_type_names = [str(t) for t in expected_type]
            if actual_type_name not in expected_type_names:
                mismatches[col] = (expected_type, actual_type_name)
        else:
            expected_type_name = str(expected_type)
            if actual_type_name != expected_type_name:
                mismatches[col] = (expected_type, actual_type_name)

    return mismatches


def generate_cast_expressions(
    df: pl.DataFrame, desired_types: dict[str, str], with_quotes: bool = True
) -> dict[str, str]:
    """Generate SQL CAST expressions for columns.

    Args:
        df: DataFrame to generate casts for
        desired_types: Dict mapping column names to SQL type names
        with_quotes: Whether to include quotes in column names

    Returns:
        Dict mapping column names to CAST expressions
    """
    cast_expressions = {}
    quote = '"' if with_quotes else ""

    for col, type_name in desired_types.items():
        if col in df.columns:
            cast_expressions[col] = f"CAST({quote}{col}{quote} AS {type_name})"

    return cast_expressions
