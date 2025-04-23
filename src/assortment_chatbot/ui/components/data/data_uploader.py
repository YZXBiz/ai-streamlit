"""
Component for handling data uploads in the assortment_chatbot.

This module provides a file uploader widget that supports multiple file formats
and handles the processing of uploaded data files.
"""

import os
import pandas as pd
import streamlit as st
import tempfile
import uuid

from assortment_chatbot.config.constants import DATA_CONFIG, FEATURES
from assortment_chatbot.ui.components.data.validation import detect_encoding, validate_file_upload
from assortment_chatbot.utils.logging import get_logger

logger = get_logger(__name__)


def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to disk in a temp directory.
    
    Parameters
    ----------
    uploaded_file : UploadedFile
        The file uploaded through Streamlit's file_uploader
        
    Returns
    -------
    str
        Path to the saved file
    """
    # Create a data directory if it doesn't exist
    data_dir = os.path.join("data", "temp")
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate a unique filename to avoid collisions
    file_extension = uploaded_file.name.split(".")[-1].lower()
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{unique_id}_{uploaded_file.name}"
    file_path = os.path.join(data_dir, filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    logger.info(f"Saved uploaded file to {file_path}")
    return file_path


def data_uploader() -> tuple[pd.DataFrame | None, str | None]:
    """
    Creates a file uploader widget for CSV, Excel, and JSON data files.
    
    This function now saves uploaded files to disk and stores the path in
    session state to enable direct file loading into DuckDB, avoiding
    Arrow compatibility issues.

    Parameters
    ----------
    None

    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        A tuple containing:
        - DataFrame with the uploaded data (None if no upload)
        - Filename of the uploaded file (None if no upload)

    Notes
    -----
    Supported file formats:
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    - JSON (.json)
    - Parquet (.parquet)

    The component automatically detects the file type from the extension
    and uses the appropriate pandas reader function.

    Examples
    --------
    >>> df, filename = data_uploader()
    >>> if df is not None:
    ...     # Do something with the DataFrame
    ...     print(f"Loaded {filename} with {len(df)} rows")
    """
    st.subheader("Data Upload")

    with st.expander("Upload your data", expanded=True):
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=[ext.replace(".", "") for ext in DATA_CONFIG["allowed_extensions"]],
            help=f"Maximum file size: {DATA_CONFIG['max_file_size_mb']}MB",
        )

        if uploaded_file is None:
            return None, None

        try:
            # Validate file - pass the Streamlit UploadedFile object directly
            is_valid, error_msg = validate_file_upload(uploaded_file)
            if not is_valid:
                st.error(error_msg)
                return None, None

            # Detect encoding - pass the Streamlit UploadedFile object directly
            encoding = detect_encoding(uploaded_file)
            
            # Save the uploaded file to disk for direct DuckDB loading
            file_path = save_uploaded_file(uploaded_file)
            
            # Store the file path in session state for the AssortmentAnalyst to use
            st.session_state.file_path = file_path
            st.session_state.file_name = uploaded_file.name

            # Load data based on file type (for preview only)
            file_extension = uploaded_file.name.lower().split(".")[-1]

            # Reset file pointer to beginning before reading
            uploaded_file.seek(0)

            if file_extension == "csv":
                df = pd.read_csv(
                    uploaded_file, encoding=encoding, nrows=DATA_CONFIG["preview_rows"]
                )
            elif file_extension in ["xlsx", "xls"]:
                df = pd.read_excel(uploaded_file, nrows=DATA_CONFIG["preview_rows"])
            elif file_extension == "json":
                df = pd.read_json(uploaded_file)
                if len(df) > DATA_CONFIG["preview_rows"]:
                    df = df.head(DATA_CONFIG["preview_rows"])
            elif file_extension == "parquet":
                df = pd.read_parquet(uploaded_file)
                if len(df) > DATA_CONFIG["preview_rows"]:
                    df = df.head(DATA_CONFIG["preview_rows"])
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None, None
                
            # Also store the preview DataFrame in session state (for backward compatibility)
            st.session_state.user_data = df

            # Show preview info
            st.success(
                f"Successfully loaded {uploaded_file.name}: {len(df)} rows × {df.shape[1]} columns"
            )
            st.info("File saved locally for direct database loading")

            # Show export option if enabled
            if FEATURES["enable_data_export"]:
                st.download_button(
                    "Download processed data",
                    df.to_csv(index=False).encode("utf-8"),
                    f"processed_{uploaded_file.name}",
                    "text/csv",
                    key="download-csv",
                )

            return df, uploaded_file.name

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            import traceback

            st.error(traceback.format_exc())
            return None, None

    return None, None
