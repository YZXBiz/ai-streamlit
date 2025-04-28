"""
Sidebar component for the PandasAI Streamlit application.

This module provides the sidebar UI for data management and application settings.
"""

import os

import pandas as pd
import pandasai as pai
import streamlit as st

from backend.app.adapters.db_sources import SQLSource
from backend.app.adapters.file_sources import CSVSource, ExcelSource, ParquetSource
from frontend.components.session_manager import render_session_manager


def render_sidebar():
    """
    Render the sidebar for data management and settings.

    This function handles:
    - File uploads
    - Database connections
    - Chat session management
    - Application settings
    """
    # Add application title
    st.sidebar.title("🐼 PandasAI")
    st.sidebar.caption("Data Analysis Assistant")

    # Render session management controls
    render_session_manager()

    # Add a separator
    st.sidebar.divider()

    with st.sidebar:
        st.markdown("<div class='sidebar-header'>Data Sources</div>", unsafe_allow_html=True)

        # File upload section
        with st.expander("Upload Data", expanded=True):
            uploaded_file = st.file_uploader(
                "Upload CSV/Parquet/Excel file", type=["csv", "parquet", "xlsx", "xls"]
            )

            if uploaded_file is not None:
                # Get file extension
                file_extension = uploaded_file.name.split(".")[-1].lower()

                # Save the file temporarily
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Create appropriate data source based on file type
                if file_extension == "csv":
                    source_class = CSVSource
                elif file_extension == "parquet":
                    source_class = ParquetSource
                elif file_extension in ["xlsx", "xls"]:
                    # For Excel, we need to handle sheet selection
                    sheet_name = st.text_input("Sheet name (leave blank for first sheet)", "")
                    if not sheet_name:
                        sheet_name = None

                # Data source name and description
                source_name = st.text_input("Dataset name", uploaded_file.name.split(".")[0])
                source_description = st.text_area(
                    "Dataset description", f"Uploaded {file_extension} file"
                )

                # Load button
                if st.button("Load Data"):
                    with st.spinner(f"Loading {uploaded_file.name}..."):
                        try:
                            # Create and load the data source
                            if file_extension in ["xlsx", "xls"]:
                                source = source_class(
                                    temp_file_path,
                                    source_name,
                                    source_description,
                                    sheet_name=sheet_name,
                                )
                            else:
                                source = source_class(
                                    temp_file_path, source_name, source_description
                                )

                            # Load the data into a PandasAI DataFrame
                            df = source.load()

                            # Register with the analyzer
                            st.session_state.analyzer.dataframe_manager.register_dataframe(
                                df, source_name, source_description
                            )

                            # Store in session state
                            if source_name not in st.session_state.loaded_dataframes:
                                st.session_state.loaded_dataframes.append(source_name)

                            st.success(f"Successfully loaded {source_name}")

                            # Clean up temporary file
                            try:
                                os.remove(temp_file_path)
                            except:
                                pass

                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")

        # Database connection section
        with st.expander("Connect to Database"):
            db_type = st.selectbox("Database Type", ["SQLite", "PostgreSQL", "MySQL"])

            if db_type == "SQLite":
                db_path = st.text_input("Database Path")

                if st.button("Connect") and db_path:
                    try:
                        connection_string = f"sqlite:///{db_path}"
                        query = st.text_area("SQL Query", "SELECT * FROM your_table LIMIT 10")
                        source_name = st.text_input("Dataset name", "sqlite_data")

                        if st.button("Execute Query"):
                            with st.spinner("Executing query..."):
                                source = SQLSource(connection_string, query, source_name)
                                df = source.load()

                                # Register with the analyzer
                                st.session_state.analyzer.dataframe_manager.register_dataframe(
                                    df, source_name
                                )

                                if source_name not in st.session_state.loaded_dataframes:
                                    st.session_state.loaded_dataframes.append(source_name)

                                st.success(f"Successfully loaded {source_name}")
                    except Exception as e:
                        st.error(f"Connection error: {str(e)}")
            else:
                st.info(f"{db_type} connection will be implemented soon")

        # Sample data option
        with st.expander("Load Sample Data"):
            if st.button("Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    # Create sample data
                    sample_data = pd.DataFrame(
                        {
                            "country": [
                                "United States",
                                "United Kingdom",
                                "France",
                                "Germany",
                                "Italy",
                                "Spain",
                                "Canada",
                                "Australia",
                                "Japan",
                                "China",
                            ],
                            "revenue": [5000, 3200, 2900, 4100, 2300, 2100, 2500, 2600, 4500, 7000],
                        }
                    )

                    # Convert to PandasAI DataFrame
                    df = pai.DataFrame(
                        sample_data,
                        name="sample_data",
                        description="Sample revenue data by country",
                    )

                    # Register with the analyzer
                    st.session_state.analyzer.dataframe_manager.register_dataframe(
                        df, "sample_data", "Sample revenue data by country"
                    )

                    # Store in session state
                    if "sample_data" not in st.session_state.loaded_dataframes:
                        st.session_state.loaded_dataframes.append("sample_data")

                    st.success("Successfully loaded sample data")

        # Display loaded dataframes
        if st.session_state.loaded_dataframes:
            st.markdown("<div class='sidebar-header'>Loaded Datasets</div>", unsafe_allow_html=True)
            for df_name in st.session_state.loaded_dataframes:
                st.markdown(f"- **{df_name}**")

            # Clear data button
            if st.button("Clear All Data"):
                from frontend.utils.session import reset_session_state

                reset_session_state()
                st.rerun()
