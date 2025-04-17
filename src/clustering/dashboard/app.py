"""Assortment-Clustering Data Explorer Application.

A streamlined application for exploring and visualizing assortment clustering data.
This application allows users to upload data files and explore them visually
using a drag-and-drop interface.
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from clustering.dashboard.components.pygwalker_view import get_pyg_renderer

# Set page configuration
st.set_page_config(
    page_title="Assortment-Clustering Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(file_path: Path | str, file_type: str | None = None) -> pd.DataFrame:
    """Load data from file with caching.

    Args:
        file_path: Path to the data file
        file_type: Type of file (csv, excel, pickle)

    Returns:
        DataFrame loaded from file
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Determine file type from extension if not provided
    if file_type is None:
        suffix = file_path.suffix.lower()
        if suffix in (".csv", ".tsv", ".txt"):
            file_type = "csv"
        elif suffix in (".xls", ".xlsx"):
            file_type = "excel"
        elif suffix == ".pkl":
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


def main():
    """Run the Assortment-Clustering Explorer application."""
    st.title("Assortment-Clustering Data Explorer")

    # Add a description of the application
    st.markdown(
        """
        Explore and visualize your clustering data using a powerful interactive interface.
        Upload a data file (CSV, Excel, or Pickle) or use the sample dataset to begin.
        """
    )

    # Sidebar for file upload and options
    with st.sidebar:
        st.header("Data Input")

        # File upload
        uploaded_file = st.file_uploader(
            "Upload your data file",
            type=["csv", "xlsx", "xls", "pkl"],
            help="Upload a CSV, Excel, or Pickle file",
        )

        # Sample dataset option
        use_sample = st.checkbox("Use sample dataset", value=not bool(uploaded_file))

        # Chart height control
        chart_height = st.slider(
            "Chart Height",
            min_value=400,
            max_value=1000,
            value=600,
            step=50,
        )

        # View selection
        view_mode = st.radio(
            "View Mode",
            ["Explorer", "Chart", "Data Profiling", "Data Preview"],
            index=0,
        )

    # Initialize session state for data if not exists
    if "data" not in st.session_state:
        st.session_state.data = None

    # Load data
    if uploaded_file is not None:
        # Save uploaded file to temp location
        temp_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Load the data
        st.session_state.data = load_data(temp_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    elif use_sample:
        # Use sample dataset
        st.session_state.data = pd.DataFrame(
            {
                "Category": ["A", "B", "C", "A", "B", "C", "A", "B", "C"] * 10,
                "Value": [10, 20, 30, 40, 50, 60, 70, 80, 90] * 10,
                "Date": pd.date_range(start="2023-01-01", periods=90, freq="D"),
                "Region": [
                    "North",
                    "South",
                    "East",
                    "West",
                    "North",
                    "South",
                    "East",
                    "West",
                    "North",
                ]
                * 10,
                "Metric": [100, 200, 300, 400, 500, 600, 700, 800, 900] * 10,
            }
        )

    # Display PyGWalker if data is loaded
    if st.session_state.data is not None:
        # Get cached renderer
        renderer = get_pyg_renderer(st.session_state.data)

        if renderer:
            # Render selected view
            if view_mode == "Explorer":
                renderer.explorer(height=chart_height)
            elif view_mode == "Chart":
                renderer.chart(height=chart_height)
            elif view_mode == "Data Profiling":
                renderer.data_profiling(height=chart_height)
            elif view_mode == "Data Preview":
                st.dataframe(st.session_state.data, use_container_width=True, height=chart_height)
    else:
        st.info("Please upload a data file or use the sample dataset to begin exploring.")

    # Footer with version info
    st.sidebar.markdown("---")
    try:
        import pygwalker

        version = pygwalker.__version__
        st.sidebar.info(f"Assortment-Clustering Explorer v1.0 (PyGWalker v{version})")
    except (ImportError, AttributeError):
        st.sidebar.warning("PyGWalker not installed. Install with `pip install pygwalker`")


if __name__ == "__main__":
    main()
