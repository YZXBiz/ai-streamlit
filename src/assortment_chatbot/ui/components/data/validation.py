"""
Validation utilities for data uploads.

This module provides functions for validating uploaded data files,
checking file types, sizes, and detecting file encodings.
"""

from typing import Any

import chardet
import pandas as pd

from assortment_chatbot.config.constants import DATA_CONFIG


def validate_file_upload(uploaded_file: Any) -> tuple[bool, str]:
    """
    Validates an uploaded file against configuration requirements.

    Parameters
    ----------
    uploaded_file : Any
        The uploaded file object from Streamlit's file_uploader

    Returns
    -------
    Tuple[bool, str]
        A tuple containing:
        - Boolean indicating if the file is valid
        - Error message if invalid, empty string if valid
    """
    if uploaded_file is None:
        return False, "No file was uploaded"

    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert bytes to MB
    max_size = DATA_CONFIG["max_file_size_mb"]
    if file_size_mb > max_size:
        return False, f"File size exceeds the maximum allowed size of {max_size}MB"

    # Check file extension
    file_extension = f".{uploaded_file.name.split('.')[-1].lower()}"
    if file_extension not in DATA_CONFIG["allowed_extensions"]:
        return (
            False,
            f"Unsupported file type: {file_extension}. Supported types: {', '.join(DATA_CONFIG['allowed_extensions'])}",
        )

    return True, ""


def detect_encoding(uploaded_file: Any) -> str:
    """
    Detects the encoding of an uploaded text file.

    Parameters
    ----------
    uploaded_file : Any
        The uploaded file object from Streamlit's file_uploader

    Returns
    -------
    str
        The detected encoding, defaults to 'utf-8' if detection fails
    """
    # Save original position
    current_position = uploaded_file.tell()

    # Read a sample of the file for encoding detection
    sample = uploaded_file.read(min(uploaded_file.size, 10000))

    # Reset to original position
    uploaded_file.seek(current_position)

    # Detect encoding
    if isinstance(sample, bytes):
        result = chardet.detect(sample)
        encoding = result["encoding"] or "utf-8"
    else:
        encoding = "utf-8"  # Default if sample is not bytes

    return encoding


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, str, pd.DataFrame]:
    """
    Validates and cleans a pandas DataFrame based on configuration rules.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate

    Returns
    -------
    Tuple[bool, str, pd.DataFrame]
        A tuple containing:
        - Boolean indicating if the DataFrame is valid
        - Message with validation results
        - The cleaned DataFrame
    """
    if df is None or df.empty:
        return False, "DataFrame is empty", df

    # Check row count
    if len(df) > DATA_CONFIG["max_rows"]:
        msg = f"DataFrame has {len(df)} rows, which exceeds the maximum of {DATA_CONFIG['max_rows']}. Sample taken."
        df = df.sample(n=DATA_CONFIG["max_rows"], random_state=42)
        return True, msg, df

    # Check for and handle missing values if configured
    missing_values = df.isna().sum().sum()
    if missing_values > 0:
        if DATA_CONFIG["handle_missing_values"]:
            # Apply configured handling strategy
            if DATA_CONFIG["missing_values_strategy"] == "drop":
                original_len = len(df)
                df = df.dropna()
                return True, f"Dropped {original_len - len(df)} rows with missing values", df
            elif DATA_CONFIG["missing_values_strategy"] == "fill":
                df = df.fillna(DATA_CONFIG["fill_value"])
                return True, f"Filled {missing_values} missing values", df
        else:
            # Just inform about missing values
            return True, f"DataFrame contains {missing_values} missing values", df

    return True, f"DataFrame validated: {len(df)} rows × {df.shape[1]} columns", df
