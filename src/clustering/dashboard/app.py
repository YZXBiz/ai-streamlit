"""Assortment-Clustering Data Explorer Application.

A streamlined application for exploring and visualizing assortment clustering data.
This application allows users to upload data files or connect to Snowflake to explore data visually.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# For data visualization
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error(
        "Plotly Express is required for enhanced visualizations. Please install it with: pip install plotly"
    )
    px = None
    go = None
    make_subplots = None

from clustering.dashboard.components.pygwalker_view import get_pyg_renderer
from clustering.io.readers import SnowflakeReader

# Custom styling for a more professional look
st.set_page_config(
    page_title="Assortment-Clustering Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    h1 {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
    }
    h2 {
        color: #1E3A8A;
        font-size: 1.8rem !important;
        padding-top: 1rem;
    }
    h3 {
        color: #3B82F6;
        font-size: 1.4rem !important;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-2px);
    }
    div.stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div.stDataFrame td {
        text-align: left;
    }
    .css-18e3th9 {
        padding: 1rem 3rem 10rem;
    }
    .st-emotion-cache-u8hs99 {
        padding: 2rem;
    }
    /* Card style for sections */
    .card {
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    /* Metric styling */
    .metric-container {
        background-color: #f8fafc;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    /* Hide hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
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


def display_df_summary(df: pd.DataFrame) -> None:
    """Display a summary of the DataFrame with visually appealing metrics.

    Args:
        df: DataFrame to summarize
    """
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Data Preview**")
        st.dataframe(df.head(5), use_container_width=True)

    with col2:
        st.write("**Data Info**")
        buffer = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Null Count": df.isna().sum(),
            }
        )
        st.dataframe(buffer, use_container_width=True)


def create_metric_card(title: str, value: Any, prefix: str = "", suffix: str = "") -> None:
    """Create a styled metric card.

    Args:
        title: Title of the metric
        value: Value to display
        prefix: Prefix to display before the value (e.g., "$")
        suffix: Suffix to display after the value (e.g., "%")
    """
    st.markdown(
        f"""
    <div class="metric-container">
        <div class="metric-value">{prefix}{value}{suffix}</div>
        <div class="metric-label">{title}</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def plot_numerical_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a distribution plot for a numerical column.

    Args:
        df: DataFrame containing the data
        column: Name of the numerical column to plot

    Returns:
        Plotly figure object with the distribution plot
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Distribution", "Box Plot"),
        specs=[[{"type": "histogram"}, {"type": "box"}]],
    )

    # Histogram
    fig.add_trace(go.Histogram(x=df[column], name=column, marker_color="#3B82F6"), row=1, col=1)

    # Box plot
    fig.add_trace(go.Box(y=df[column], name=column, marker_color="#3B82F6"), row=1, col=2)

    fig.update_layout(
        title=f"Distribution Analysis: {column}",
        height=400,
        showlegend=False,
        template="plotly_white",
    )

    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Create a correlation matrix heatmap.

    Args:
        df: DataFrame containing numerical data

    Returns:
        Plotly figure object with the correlation heatmap
    """
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns, colorscale="Blues", zmin=-1, zmax=1
        )
    )

    fig.update_layout(title="Correlation Matrix", height=600, width=700, template="plotly_white")

    return fig


def main():
    """Run the Assortment-Clustering Explorer application."""
    # Header with logo and title in a modern layout
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown(
            """
        <div style='text-align: center; padding: 10px;'>
            <span style='font-size: 3rem; color: #4F46E5;'>📊</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <h1 style='margin-bottom: 0;'>Assortment-Clustering Explorer</h1>
        <p style='color: #6B7280; margin-top: 0;'>
            Interactive data visualization for cluster analysis and exploration
        </p>
        """,
            unsafe_allow_html=True,
        )

    # Main content area with description
    st.markdown(
        """
    <div style='background-color: #EFF6FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem;'>
        <h3 style='margin-top: 0; color: #1E40AF;'>
            <span style='margin-right: 10px;'>📊</span>
            Explore Your Data
        </h3>
        <p>
            Analyze and visualize your clustering data using a powerful, interactive interface.
            Upload a data file, connect to Snowflake, or use a sample dataset to begin your exploration journey.
        </p>
        <p>
            <strong>Getting Started:</strong> Select a data source from the sidebar and follow the prompts to
            load your data. Once loaded, use the interactive tools to create visualizations and gain insights.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar for data source options
    with st.sidebar:
        st.markdown(
            """
        <h2 style='color: #E2E8F0; margin-bottom: 1.5rem; border-bottom: 1px solid #475569; padding-bottom: 0.5rem;'>
            <span style="margin-right: 8px;">📂</span> Data Source
        </h2>
        """,
            unsafe_allow_html=True,
        )

        # Initialize variables to avoid 'before assignment' errors
        uploaded_file = None
        snowflake_query = ""
        load_sf_data = False
        use_cache = True

        # Source selection tabs with icons
        data_source = st.radio(
            "Select Data Source",
            ["📄 File Upload", "❄️ Snowflake"],
            index=0,
            format_func=lambda x: x,
        )

        if "📄 File Upload" in data_source:
            # File upload with enhanced UI
            st.markdown("### Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["csv", "xlsx", "xls", "pkl"],
                help="Upload a CSV, Excel, or Pickle file",
            )

            if uploaded_file is None:
                st.info("👆 Drag and drop your file here or click to browse")

        elif "❄️ Snowflake" in data_source:
            # Snowflake query input with cleaner UI
            st.markdown("### Query Snowflake Database")

            st.info("Enter SQL to retrieve data from Snowflake")
            snowflake_query = st.text_area(
                "SQL Query", height=150, placeholder="SELECT * FROM your_table LIMIT 1000"
            )
            use_cache = st.checkbox("💾 Use cached results (if available)", value=True)

            # More prominent load button
            load_sf_data = st.button(
                "🚀 Load Data from Snowflake", type="primary", use_container_width=True
            )

        # Visualization options with cleaner layout
        st.markdown(
            """
        <h2 style='color: #E2E8F0; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 1px solid #475569; padding-bottom: 0.5rem;'>
            <span style="margin-right: 8px;">🎨</span> Visualization Options
        </h2>
        """,
            unsafe_allow_html=True,
        )

        chart_height = st.slider(
            "Visualization Height",
            min_value=400,
            max_value=1000,
            value=600,
            step=50,
            format="%d px",
        )

        # View selection with icons and better descriptions
        st.markdown("### View Mode")
        view_mode = st.radio(
            "Select Mode",
            ["🔍 Explorer", "📊 Chart", "📋 Data Profiling", "👁️ Data Preview"],
            index=0,
            help="Choose how you want to view and interact with your data",
        )

        # Map back to the internal values
        view_mode_map = {
            "🔍 Explorer": "Explorer",
            "📊 Chart": "Chart",
            "📋 Data Profiling": "Data Profiling",
            "👁️ Data Preview": "Data Preview",
        }
        view_mode = view_mode_map[view_mode]

    # Initialize session state for data if not exists
    if "data" not in st.session_state:
        st.session_state.data = None

    # Add a loading animation for better UX
    with st.spinner("Processing data..."):
        # Load data based on selected source
        if "📄 File Upload" in data_source and uploaded_file is not None:
            # Save uploaded file to temp location
            temp_path = Path(f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Load the data
            st.session_state.data = load_data(temp_path)

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Show success message
            if not st.session_state.data.empty:
                st.success(
                    f"✅ Successfully loaded {len(st.session_state.data):,} rows from {uploaded_file.name}"
                )

        elif "❄️ Snowflake" in data_source and load_sf_data and snowflake_query:
            # Load data from Snowflake with the provided query
            with st.spinner("Connecting to Snowflake..."):
                st.session_state.data = load_snowflake_data(
                    query=snowflake_query, use_cache=use_cache
                )

                if not st.session_state.data.empty:
                    st.success(
                        f"✅ Successfully loaded {len(st.session_state.data):,} rows from Snowflake"
                    )
                else:
                    st.error("❌ No data returned from Snowflake query")

    # Display visualization if data is loaded
    if st.session_state.data is not None:
        # Show data summary in an elegant card
        st.markdown("## Dataset Overview")
        display_df_summary(st.session_state.data)

        # Horizontal line separator
        st.markdown("<hr style='margin: 2rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

        # Visualization section
        st.markdown("## Interactive Visualization")

        # Get cached renderer
        renderer = get_pyg_renderer(st.session_state.data)

        if renderer:
            # Render selected view
            if view_mode == "Explorer":
                # Use the explorer method for the standard view
                renderer.explorer()
            elif view_mode == "Chart":
                # For Chart view, we need to specify an index for which chart to display
                # Since there might not be any charts yet, we'll safely wrap this in try/except
                try:
                    # Show the first chart (index 0)
                    renderer.chart(0)
                except Exception as e:
                    st.warning("⚠️ No charts available. Create a chart in Explorer view first.")
                    st.error(f"Error: {str(e)}")
            elif view_mode == "Data Profiling":
                # For Data Profiling, we use explorer with "data" as the default tab
                renderer.explorer(default_tab="data")
            elif view_mode == "Data Preview":
                st.dataframe(st.session_state.data, use_container_width=True, height=chart_height)
    else:
        # Enhanced empty state with illustrations and guidance
        if "📄 File Upload" in data_source:
            st.markdown(
                """
            <div style='text-align: center; padding: 3rem 1rem; background-color: white; border-radius: 0.5rem; margin: 2rem 0;'>
                <div style='font-size: 4rem; margin-bottom: 1.5rem;'>📤</div>
                <h3>Upload Your Dataset to Begin</h3>
                <p style='color: #6B7280; max-width: 600px; margin: 1rem auto;'>
                    Select a CSV, Excel, or Pickle file from your device to begin exploring your cluster data.
                    Use the file uploader in the sidebar to get started.
                </p>
            </div>
            """,
                unsafe_allow_html=True,
            )
        elif "❄️ Snowflake" in data_source:
            st.markdown(
                """
            <div style='text-align: center; padding: 3rem 1rem; background-color: white; border-radius: 0.5rem; margin: 2rem 0;'>
                <div style='font-size: 4rem; margin-bottom: 1.5rem;'>❄️</div>
                <h3>Connect to Snowflake</h3>
                <p style='color: #6B7280; max-width: 600px; margin: 1rem auto;'>
                    Enter your SQL query in the sidebar and click "Load Data from Snowflake" to begin exploring your data.
                    For best performance, add limits to your query.
                </p>
                <div style='background-color: #F3F4F6; padding: 1rem; border-radius: 0.3rem; margin-top: 1.5rem; text-align: left; max-width: 600px; margin-left: auto; margin-right: auto;'>
                    <code>SELECT * FROM your_table WHERE cluster_id IS NOT NULL LIMIT 1000</code>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    # Enhanced footer with version info and credits
    st.sidebar.markdown("<hr style='margin: 2rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

    st.sidebar.markdown(
        """
    <div style='text-align: center; padding: 1rem 0;'>
        <p style='color: #94A3B8; margin-bottom: 0.5rem;'>Assortment-Clustering Explorer v1.0</p>
        <p style='color: #94A3B8; font-size: 0.8rem; margin-top: 0;'>
            Built with ❤️ using Streamlit & PyGWalker
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
