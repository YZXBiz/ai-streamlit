"""Data Upload Page for Clustering Dashboard.

This page allows users to upload data files or connect to Snowflake.
"""

import pandas as pd
import streamlit as st

from clustering.shared.utils.io.readers import SnowflakeReader


@st.cache_data
def load_data(file_path, file_type=None):
    """Load data from file with caching.

    Args:
        file_path: Path to the data file
        file_type: Type of file (csv, excel, pickle)

    Returns:
        DataFrame loaded from file
    """
    # Determine file type from extension if not provided
    if file_type is None:
        suffix = str(file_path).lower()
        if suffix.endswith((".csv", ".tsv", ".txt")):
            file_type = "csv"
        elif suffix.endswith((".xls", ".xlsx")):
            file_type = "excel"
        elif suffix.endswith(".pkl"):
            file_type = "pickle"
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    # Load data based on file type
    try:
        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "excel":
            return pd.read_excel(file_path)
        elif file_type == "pickle":
            return pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def load_snowflake_data(query: str, use_cache: bool = True) -> pd.DataFrame:
    """Load data from Snowflake with caching.

    Args:
        query: SQL query to execute against Snowflake
        use_cache: Whether to use local caching for the query

    Returns:
        DataFrame loaded from Snowflake
    """
    try:
        # Use the SnowflakeReader from clustering.io
        reader = SnowflakeReader(query=query, use_cache=use_cache)

        # Read data using the reader
        df = reader.read()

        # Convert from polars to pandas DataFrame
        return df.to_pandas()
    except Exception as e:
        st.error(f"Error loading data from Snowflake: {str(e)}")
        return pd.DataFrame()


def data_upload_page():
    """Render the data upload page."""
    st.title("🗂️ Data Upload")

    # Create tabs for different data sources
    source_tabs = st.tabs(["📁 File Upload", "❄️ Snowflake"])

    # File Upload tab
    with source_tabs[0]:
        st.header("Upload a Data File")

        uploaded_file = st.file_uploader(
            "Choose a CSV, Excel, or Pickle file", type=["csv", "xlsx", "xls", "pkl"]
        )

        if uploaded_file is not None:
            # Load the data
            df = load_data(uploaded_file)

            # Store in session state for use in other pages
            st.session_state["data"] = df
            st.session_state["data_source"] = "file"

            # Show success message
            st.success(
                f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns"
            )

            # Preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)

    # Snowflake tab
    with source_tabs[1]:
        st.header("Connect to Snowflake")

        # Input for SQL query
        query = st.text_area(
            "Enter SQL Query", height=150, placeholder="SELECT * FROM your_table LIMIT 1000"
        )

        # Option to use cache
        use_cache = st.checkbox("Use cached data (if available)", value=True)

        if st.button("Execute Query"):
            if query:
                # Show loading spinner
                with st.spinner("Executing query..."):
                    df = load_snowflake_data(query, use_cache)

                if not df.empty:
                    # Store in session state for use in other pages
                    st.session_state["data"] = df
                    st.session_state["data_source"] = "snowflake"

                    # Show success message
                    st.success(
                        f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns"
                    )

                    # Preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
            else:
                st.warning("Please enter a SQL query")


# Run the page
data_upload_page()
