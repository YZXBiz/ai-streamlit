"""
Component for handling data uploads in the assortment_chatbot.

This module provides a file uploader widget that supports multiple file formats
and handles the processing of uploaded data files.
"""

import pandas as pd
import streamlit as st

from assortment_chatbot.config.constants import DATA_CONFIG, FEATURES
from assortment_chatbot.ui.components.data.validation import detect_encoding, validate_file_upload


def data_uploader() -> tuple[pd.DataFrame | None, str | None]:
    """
    Creates a file uploader widget for CSV, Excel, and JSON data files.

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

            # Load data based on file type
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
            else:
                st.error(f"Unsupported file type: {file_extension}")
                return None, None

            # Show preview info
            st.success(
                f"Successfully loaded {uploaded_file.name}: {len(df)} rows × {df.shape[1]} columns"
            )

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
