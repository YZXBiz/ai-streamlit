"""
File utility functions for assortment_chatbot data handling.

This module provides utility functions for handling file operations
such as reading various file formats, saving data, and exporting results.
"""

import base64
import io
import json
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd


def read_csv_file(file: BinaryIO, encoding: str = "utf-8", **kwargs) -> pd.DataFrame:
    """
    Read a CSV file into a pandas DataFrame.

    Args:
        file: The file object to read
        encoding: The file encoding (default: utf-8)
        **kwargs: Additional arguments to pass to pd.read_csv

    Returns:
        DataFrame containing the CSV data
    """
    try:
        return pd.read_csv(file, encoding=encoding, **kwargs)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")


def read_excel_file(file: BinaryIO, **kwargs) -> pd.DataFrame:
    """
    Read an Excel file into a pandas DataFrame.

    Args:
        file: The file object to read
        **kwargs: Additional arguments to pass to pd.read_excel

    Returns:
        DataFrame containing the Excel data
    """
    try:
        return pd.read_excel(file, **kwargs)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {str(e)}")


def read_json_file(file: BinaryIO, **kwargs) -> pd.DataFrame:
    """
    Read a JSON file into a pandas DataFrame.

    Args:
        file: The file object to read
        **kwargs: Additional arguments to pass to pd.read_json

    Returns:
        DataFrame containing the JSON data
    """
    try:
        return pd.read_json(file, **kwargs)
    except Exception as e:
        raise ValueError(f"Error reading JSON file: {str(e)}")


def read_file(
    file: BinaryIO, file_extension: str, encoding: str = "utf-8", **kwargs
) -> pd.DataFrame:
    """
    Read a file into a pandas DataFrame based on its extension.

    Args:
        file: The file object to read
        file_extension: The file extension (e.g., ".csv")
        encoding: The file encoding for text files (default: utf-8)
        **kwargs: Additional arguments to pass to the reader function

    Returns:
        DataFrame containing the file data

    Raises:
        ValueError: If the file format is not supported
    """
    file_extension = file_extension.lower()

    if file_extension in [".csv"]:
        return read_csv_file(file, encoding=encoding, **kwargs)
    elif file_extension in [".xlsx", ".xls"]:
        return read_excel_file(file, **kwargs)
    elif file_extension in [".json"]:
        return read_json_file(file, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def save_dataframe(df: pd.DataFrame, filename: str, format_type: str = "csv") -> tuple[bool, str]:
    """
    Save a DataFrame to the specified format.

    Args:
        df: The DataFrame to save
        filename: The name of the file (without extension)
        format_type: The format to save the file as (csv, excel, json)

    Returns:
        Tuple of (success, message)
    """
    try:
        output_dir = Path("data/output")
        output_dir.mkdir(exist_ok=True, parents=True)

        filepath = output_dir / f"{filename}.{format_type}"

        if format_type == "csv":
            df.to_csv(filepath, index=False)
        elif format_type == "excel":
            df.to_excel(filepath, index=False)
        elif format_type == "json":
            df.to_json(filepath, orient="records")
        else:
            return False, f"Unsupported format: {format_type}"

        return True, f"File saved successfully to {filepath}"
    except Exception as e:
        return False, f"Error saving file: {str(e)}"


def get_download_link(df: pd.DataFrame, filename: str, format_type: str = "csv") -> str:
    """
    Generate a download link for a DataFrame.

    Args:
        df: The DataFrame to download
        filename: The name of the file (without extension)
        format_type: The format to download (csv, excel, json)

    Returns:
        HTML string containing the download link
    """
    try:
        if format_type == "csv":
            data = df.to_csv(index=False)
            b64 = base64.b64encode(data.encode()).decode()
            mime = "text/csv"
            extension = "csv"
        elif format_type == "excel":
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False)
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode()
            mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            extension = "xlsx"
        elif format_type == "json":
            data = df.to_json(orient="records")
            b64 = base64.b64encode(data.encode()).decode()
            mime = "application/json"
            extension = "json"
        else:
            return "Unsupported format type"

        href = f'<a href="data:{mime};base64,{b64}" download="{filename}.{extension}">Download {filename}.{extension}</a>'
        return href
    except Exception as e:
        return f"Error generating download link: {str(e)}"


def load_json_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Dictionary containing the configuration data

    Raises:
        FileNotFoundError: If the configuration file does not exist
        ValueError: If the configuration file is not valid JSON
    """
    try:
        with open(config_path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in configuration file: {config_path}")
