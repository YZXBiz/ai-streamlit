"""Data validation utilities for file uploads and data processing."""

import os
import io
from typing import Tuple, Union, BinaryIO
from dashboard.constants import DATA_CONFIG

def validate_file_upload(file_obj: Union[str, BinaryIO, "UploadedFile"]) -> Tuple[bool, str]:
    """Validate uploaded file against configuration rules.
    
    Args:
        file_obj: Either a file path string or a Streamlit UploadedFile object
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    # Handle different types of file objects
    if isinstance(file_obj, str):
        # It's a file path
        file_name = file_obj
        # Check file extension
        _, ext = os.path.splitext(file_name.lower())
        if ext not in DATA_CONFIG["allowed_extensions"]:
            return False, f"Unsupported file type. Allowed types: {', '.join(DATA_CONFIG['allowed_extensions'])}"
        
        # Check file size if it's a path
        try:
            file_size_mb = os.path.getsize(file_obj) / (1024 * 1024)
            if file_size_mb > DATA_CONFIG["max_file_size_mb"]:
                return False, f"File too large. Maximum size: {DATA_CONFIG['max_file_size_mb']}MB"
        except (FileNotFoundError, PermissionError):
            # Don't fail on file system errors - we'll handle the file content later
            pass
    else:
        # It's a Streamlit UploadedFile or file-like object
        try:
            file_name = getattr(file_obj, "name", "uploaded_file")
            # Check file extension
            _, ext = os.path.splitext(file_name.lower())
            if ext not in DATA_CONFIG["allowed_extensions"]:
                return False, f"Unsupported file type. Allowed types: {', '.join(DATA_CONFIG['allowed_extensions'])}"
            
            # Check file size if size attribute is available
            if hasattr(file_obj, "size"):
                file_size_mb = file_obj.size / (1024 * 1024)
                if file_size_mb > DATA_CONFIG["max_file_size_mb"]:
                    return False, f"File too large. Maximum size: {DATA_CONFIG['max_file_size_mb']}MB"
        except Exception as e:
            return False, f"Error validating file: {str(e)}"
    
    return True, ""

def detect_encoding(file_obj: Union[str, BinaryIO, "UploadedFile"]) -> str:
    """Detect file encoding from supported encodings.
    
    Args:
        file_obj: Either a file path string or a Streamlit UploadedFile object
        
    Returns:
        str: Detected encoding
    """
    # Handle file path case
    if isinstance(file_obj, str):
        for encoding in DATA_CONFIG["supported_encodings"]:
            try:
                with open(file_obj, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try reading first 1024 bytes
                    return encoding
            except (UnicodeDecodeError, FileNotFoundError, PermissionError):
                continue
    else:
        # Handle Streamlit UploadedFile or file-like object
        # Save the current position to restore it later
        if hasattr(file_obj, "tell") and hasattr(file_obj, "seek"):
            current_position = file_obj.tell()
        else:
            current_position = None
            
        # Try to detect encoding
        for encoding in DATA_CONFIG["supported_encodings"]:
            try:
                # Reset to beginning of file if possible
                if current_position is not None:
                    file_obj.seek(0)
                    
                # Read a sample and check if it can be decoded
                if hasattr(file_obj, "read"):
                    sample = file_obj.read(1024)
                    # Convert bytes to string using the encoding
                    if isinstance(sample, bytes):
                        sample.decode(encoding)
                        
                    # Reset to beginning of file again
                    if current_position is not None:
                        file_obj.seek(0)
                        
                    return encoding
            except (UnicodeDecodeError, AttributeError, ValueError):
                continue
        
        # Restore the original position if possible
        if current_position is not None:
            try:
                file_obj.seek(current_position)
            except (AttributeError, ValueError):
                pass
    
    # Default to first supported encoding if detection fails
    return DATA_CONFIG["supported_encodings"][0] 