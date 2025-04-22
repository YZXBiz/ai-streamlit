"""
Data conversion utilities.

This module provides functions for converting between different data formats.
"""
import io
import pandas as pd
from typing import Optional, Dict, Any, Union, BinaryIO

def dataframe_to_csv(df: pd.DataFrame) -> str:
    """
    Convert DataFrame to CSV string.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        CSV string
    """
    return df.to_csv(index=False)

def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to Excel bytes.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

def dataframe_to_json(df: pd.DataFrame, orient: str = "records") -> str:
    """
    Convert DataFrame to JSON string.
    
    Args:
        df: DataFrame to convert
        orient: JSON orientation (records, split, index, columns, values)
        
    Returns:
        JSON string
    """
    return df.to_json(orient=orient)

def dataframe_to_parquet(df: pd.DataFrame) -> bytes:
    """
    Convert DataFrame to Parquet bytes.
    
    Args:
        df: DataFrame to convert
        
    Returns:
        Parquet file as bytes
    """
    output = io.BytesIO()
    df.to_parquet(output, index=False)
    return output.getvalue()

def csv_to_dataframe(csv_data: Union[str, bytes, BinaryIO]) -> pd.DataFrame:
    """
    Convert CSV to DataFrame.
    
    Args:
        csv_data: CSV data as string, bytes, or file-like object
        
    Returns:
        DataFrame
    """
    return pd.read_csv(csv_data)

def excel_to_dataframe(excel_data: Union[bytes, BinaryIO], sheet_name: Optional[str] = None) -> pd.DataFrame:
    """
    Convert Excel to DataFrame.
    
    Args:
        excel_data: Excel data as bytes or file-like object
        sheet_name: Sheet name to read (default: first sheet)
        
    Returns:
        DataFrame
    """
    return pd.read_excel(excel_data, sheet_name=sheet_name)

def json_to_dataframe(json_data: Union[str, bytes, BinaryIO], orient: str = "records") -> pd.DataFrame:
    """
    Convert JSON to DataFrame.
    
    Args:
        json_data: JSON data as string, bytes, or file-like object
        orient: JSON orientation (records, split, index, columns, values)
        
    Returns:
        DataFrame
    """
    return pd.read_json(json_data, orient=orient)

def parquet_to_dataframe(parquet_data: Union[bytes, BinaryIO]) -> pd.DataFrame:
    """
    Convert Parquet to DataFrame.
    
    Args:
        parquet_data: Parquet data as bytes or file-like object
        
    Returns:
        DataFrame
    """
    return pd.read_parquet(parquet_data)

def infer_and_convert(data: Union[str, bytes, BinaryIO], file_type: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Infer file type and convert to DataFrame.
    
    Args:
        data: Data as string, bytes, or file-like object
        file_type: File type hint (csv, excel, json, parquet)
        
    Returns:
        DataFrame or None if conversion fails
    """
    try:
        if file_type:
            file_type = file_type.lower()
            
            if file_type == 'csv':
                return csv_to_dataframe(data)
            elif file_type in ['xlsx', 'xls', 'excel']:
                return excel_to_dataframe(data)
            elif file_type == 'json':
                return json_to_dataframe(data)
            elif file_type == 'parquet':
                return parquet_to_dataframe(data)
            else:
                print(f"Unsupported file type: {file_type}")
                return None
        else:
            # Try different formats in order of likelihood
            try:
                return csv_to_dataframe(data)
            except Exception:
                try:
                    return json_to_dataframe(data)
                except Exception:
                    try:
                        return excel_to_dataframe(data)
                    except Exception:
                        try:
                            return parquet_to_dataframe(data)
                        except Exception:
                            print("Could not infer file type")
                            return None
    except Exception as e:
        print(f"Error converting data: {e}")
        return None 