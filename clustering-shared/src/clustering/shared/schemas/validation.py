"""DataFrame schema validation utilities."""

from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import polars as pl


def validate_dataframe_schema(
    df: Union[pd.DataFrame, pl.DataFrame],
    expected_schema: Dict[str, type],
    optional_fields: Optional[List[str]] = None,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Validate a DataFrame against an expected schema.
    
    Args:
        df: The DataFrame to validate (either pandas or polars)
        expected_schema: Dictionary mapping column names to expected types
        optional_fields: List of fields that are not required
        strict: Whether to enforce that no extra columns are present
        
    Returns:
        Dict containing validation results:
            passed: Boolean indicating if validation passed
            missing_fields: List of required fields missing from the DataFrame
            extra_fields: List of fields in DataFrame not in expected schema
            type_mismatches: List of fields with type mismatches
    """
    # Convert polars DataFrame to pandas if needed
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    
    # Get column sets
    df_columns = set(df.columns)
    expected_columns = set(expected_schema.keys())
    optional_fields_set = set(optional_fields or [])
    required_columns = expected_columns - optional_fields_set
    
    # Check for missing required fields
    missing_fields = required_columns - df_columns
    
    # Check for extra fields
    extra_fields = df_columns - expected_columns if strict else set()
    
    # Check type mismatches
    type_mismatches = []
    for col, expected_type in expected_schema.items():
        if col in df.columns:
            # Special case for strings
            if expected_type == str:
                if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
                    type_mismatches.append((col, str(df[col].dtype), str(expected_type)))
            # Special case for integers
            elif expected_type == int:
                if not pd.api.types.is_integer_dtype(df[col]) and not pd.api.types.is_float_dtype(df[col]):
                    type_mismatches.append((col, str(df[col].dtype), str(expected_type)))
            # Special case for floats
            elif expected_type == float:
                if not pd.api.types.is_float_dtype(df[col]) and not pd.api.types.is_integer_dtype(df[col]):
                    type_mismatches.append((col, str(df[col].dtype), str(expected_type)))
            # Standard type checking for other types
            elif not isinstance(df[col].dtype, expected_type):
                type_mismatches.append((col, str(df[col].dtype), str(expected_type)))
    
    # Determine if validation passed
    passed = len(missing_fields) == 0 and len(extra_fields) == 0 and len(type_mismatches) == 0
    
    return {
        "passed": passed,
        "missing_fields": list(missing_fields),
        "extra_fields": list(extra_fields),
        "type_mismatches": type_mismatches,
    }


def fix_dataframe_schema(
    df: pd.DataFrame,
    expected_schema: Dict[str, type],
    optional_fields: Optional[List[str]] = None,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Fix a DataFrame to conform to an expected schema.
    
    Args:
        df: The DataFrame to fix
        expected_schema: Dictionary mapping column names to expected types
        optional_fields: List of fields that are not required
        strict: Whether to remove extra columns not in schema
        
    Returns:
        Fixed DataFrame conforming to the schema
    """
    df_copy = df.copy()
    optional_fields = optional_fields or []
    
    # Add missing columns with default values
    for col, col_type in expected_schema.items():
        if col not in df_copy.columns:
            # Skip optional fields
            if col in optional_fields:
                continue
                
            # Add with default value based on type
            if col_type == str:
                df_copy[col] = ""
            elif col_type == int:
                df_copy[col] = 0
            elif col_type == float:
                df_copy[col] = 0.0
            elif col_type == bool:
                df_copy[col] = False
            else:
                df_copy[col] = None
    
    # Fix column types
    for col, col_type in expected_schema.items():
        if col in df_copy.columns:
            if col_type == str:
                # Convert to string
                if not pd.api.types.is_object_dtype(df_copy[col]) and not pd.api.types.is_string_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].astype(str)
            elif col_type == int:
                # Convert to integer
                if not pd.api.types.is_integer_dtype(df_copy[col]):
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0).astype(int)
            elif col_type == float:
                # Convert to float
                if not pd.api.types.is_float_dtype(df_copy[col]):
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0)
            elif col_type == bool:
                # Convert to boolean
                if not pd.api.types.is_bool_dtype(df_copy[col]):
                    df_copy[col] = df_copy[col].astype(bool)
    
    # Remove extra columns if strict
    if strict:
        expected_columns = set(expected_schema.keys())
        extra_columns = [col for col in df_copy.columns if col not in expected_columns]
        if extra_columns:
            df_copy = df_copy.drop(columns=extra_columns)
    
    return df_copy 