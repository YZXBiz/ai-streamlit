"""Assortment-Clustering Data Explorer Application.

A streamlined application for exploring and visualizing assortment clustering data.
This application allows users to upload data files or connect to Snowflake to explore data visually.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
    
    /* Hide any "Visualization Options" sections that might be added by plugins */
    [data-testid="stSidebarContent"] div:has(h2:contains("Visualization Options")) {
        display: none !important;
    }
    [data-testid="stSidebarContent"] h2:contains("Visualization Options") {
        display: none !important;
    }
    .visualization-options {
        display: none !important;
    }
    
    /* Hide the navigation header when using Snowflake */
    .snowflake-mode div:has(h2:contains("Navigation")) {
        display: none !important;
    }
    .snowflake-mode [data-testid="stSidebarContent"] h1:contains("Navigation") {
        display: none !important;
    }
    .snowflake-mode #root div:has(p:contains("Navigation")) {
        display: none !important;
    }
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
    """Display a comprehensive summary of the DataFrame with professional metrics and visualizations.

    Args:
        df: DataFrame to summarize
    """
    # Dataset top-level metrics in visually appealing cards
    st.markdown("<h3>📊 Dataset Metrics</h3>", unsafe_allow_html=True)

    # Key metrics row with 4 cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_metric_card("Total Records", f"{len(df):,}")
    with col2:
        create_metric_card("Columns", f"{df.shape[1]:,}")
    with col3:
        num_cols = len(df.select_dtypes(include=[np.number]).columns)
        create_metric_card("Numeric Columns", f"{num_cols:,}")
    with col4:
        missing_percent = round((df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
        create_metric_card("Missing Values", f"{missing_percent}%", suffix="%")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Data preview and schema in tabs
    st.markdown("<h3>🔍 Data Structure</h3>", unsafe_allow_html=True)
    data_tabs = st.tabs(["✨ Preview", "📋 Schema", "📊 Data Types"])

    with data_tabs[0]:  # Preview tab
        st.dataframe(df.head(5), use_container_width=True)

    with data_tabs[1]:  # Schema tab
        buffer = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": df.dtypes.astype(str),
                "Non-Null Count": df.count(),
                "Non-Null %": (df.count() / len(df) * 100).round(2).astype(str) + "%",
                "Null Count": df.isna().sum(),
                "Unique Values": [
                    df[col].nunique() if df[col].nunique() < 1000 else ">1000" for col in df.columns
                ],
            }
        )
        st.dataframe(buffer, use_container_width=True)

    with data_tabs[2]:  # Data Types tab
        # Count columns by data type
        type_counts = df.dtypes.value_counts().reset_index()
        type_counts.columns = ["Data Type", "Count"]

        # Create a horizontal bar chart
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=type_counts["Data Type"].astype(str),
                x=type_counts["Count"],
                orientation="h",
                marker_color="#3B82F6",
            )
        )

        fig.update_layout(
            title="Column Data Types Distribution",
            xaxis_title="Count",
            yaxis_title="Data Type",
            height=300,
            margin=dict(l=0, r=0, b=0, t=40),
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

    # Statistical summary in expandable section
    st.markdown("<h3>📈 Statistical Summary</h3>", unsafe_allow_html=True)

    # Only include numeric columns for summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if numeric_cols:
        stats_tabs = st.tabs(["Summary", "Correlation Matrix"])

        with stats_tabs[0]:
            stats_df = df[numeric_cols].describe().T
            # Add more useful statistics
            stats_df["missing"] = df[numeric_cols].isna().sum()
            stats_df["missing_pct"] = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)

            # Reorder and rename columns for clarity
            stats_df = stats_df[
                [
                    "count",
                    "missing",
                    "missing_pct",
                    "mean",
                    "std",
                    "min",
                    "25%",
                    "50%",
                    "75%",
                    "max",
                ]
            ]
            stats_df.columns = [
                "Count",
                "Missing",
                "Missing %",
                "Mean",
                "Std Dev",
                "Min",
                "25th Perc",
                "Median",
                "75th Perc",
                "Max",
            ]

            st.dataframe(stats_df, use_container_width=True)

        with stats_tabs[1]:
            if len(numeric_cols) > 1:
                fig = plot_correlation_matrix(df[numeric_cols])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns to create a correlation matrix.")
    else:
        st.info("No numeric columns found for statistical analysis.")

    # Sample of selected columns (if too many columns)
    if len(df.columns) > 8:
        st.markdown("<h3>📊 Column Highlights</h3>", unsafe_allow_html=True)

        # Select a sample of diverse columns
        numeric_sample = numeric_cols[: min(3, len(numeric_cols))]
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        categorical_sample = categorical_cols[: min(3, len(categorical_cols))]

        sample_cols = numeric_sample + categorical_sample

        if sample_cols:
            for i in range(0, len(sample_cols), 3):
                cols = st.columns(min(3, len(sample_cols) - i))
                for j, col in enumerate(cols):
                    if i + j < len(sample_cols):
                        column_name = sample_cols[i + j]
                        with col:
                            st.markdown(f"**{column_name}**")
                            if column_name in numeric_cols:
                                # For numeric columns, show a small histogram
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Histogram(x=df[column_name].dropna(), marker_color="#3B82F6")
                                )
                                fig.update_layout(
                                    height=200,
                                    margin=dict(l=0, r=0, b=0, t=0),
                                    template="plotly_white",
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                # For categorical columns, show top values
                                value_counts = df[column_name].value_counts().head(5)
                                st.write(
                                    f"Top values ({min(5, len(value_counts))} of {df[column_name].nunique()}):"
                                )
                                st.write(value_counts)


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
    # Initialize session state variables if they don't exist
    if "data" not in st.session_state:
        st.session_state.data = None

    if "data_source" not in st.session_state:
        st.session_state.data_source = "📄 File Upload"

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Data Loading"

    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Explorer"

    if "hide_navigation" not in st.session_state:
        st.session_state.hide_navigation = False

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

    # Sidebar for data source options
    with st.sidebar:
        st.markdown(
            """
        <h2 style='color: #000000; margin-bottom: 1.5rem; border-bottom: 1px solid #818CF8; padding-bottom: 0.5rem; font-weight: 600; font-size: 1.25rem;'>
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
            index=0 if st.session_state.data_source == "📄 File Upload" else 1,
            format_func=lambda x: x,
        )

        # Update data source in session state
        st.session_state.data_source = data_source

        # Add special class to body when in Snowflake mode to hide navigation
        if "❄️ Snowflake" in data_source:
            st.markdown(
                """
            <script>
                document.body.classList.add('snowflake-mode');
            </script>
            <style>
                .snowflake-mode h1:contains("Navigation"), 
                .snowflake-mode h2:contains("Navigation"),
                .snowflake-mode div:has(> h1:contains("Navigation")),
                .snowflake-mode div:has(> p:contains("View Mode")) {
                    display: none !important;
                }
            </style>
            """,
                unsafe_allow_html=True,
            )

        # Reset view when switching data sources - helps clean up UI
        if (
            "last_data_source" in st.session_state
            and st.session_state.last_data_source != data_source
        ):
            current_source = data_source
            if st.button("🔄 Reset View", help="Clear the current view and navigation"):
                for key in list(st.session_state.keys()):
                    if key != "data_source":
                        del st.session_state[key]
                st.session_state.data_source = current_source
                st.rerun()

        # Store current data source for comparison on next render
        st.session_state.last_data_source = data_source

        if "📄 File Upload" in data_source:
            # File upload with enhanced UI
            st.markdown("### Upload Your Data")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["csv", "xlsx", "xls", "pkl"],
                help="Upload a CSV, Excel, or Pickle file",
                key="data_upload",
            )

            if uploaded_file is None:
                st.info("👆 Drag and drop your file here or click to browse")

        elif "❄️ Snowflake" in data_source:
            # Snowflake query input with cleaner UI
            st.markdown("### Query Snowflake Database")

            # Pre-populate the query if it exists in session state
            query_value = ""
            if "snowflake_query" in st.session_state:
                query_value = st.session_state.snowflake_query

            st.info("Enter SQL to retrieve data from Snowflake")
            snowflake_query = st.text_area(
                "SQL Query",
                value=query_value,
                height=150,
                placeholder="SELECT * FROM your_table LIMIT 1000",
            )

            # Save query to session state
            if snowflake_query != query_value:
                st.session_state.snowflake_query = snowflake_query

            use_cache = st.checkbox("💾 Use cached results (if available)", value=True)

            # More prominent load button
            load_sf_data = st.button(
                "🚀 Load Data from Snowflake", type="primary", use_container_width=True
            )

        # Only show navigation if data is loaded AND we're not in Snowflake mode
        if (
            "data" in st.session_state
            and st.session_state.data is not None
            and "❄️ Snowflake" not in data_source
        ):
            # Add toggle button for navigation
            if st.button(
                "🔀 Hide Navigation"
                if not st.session_state.hide_navigation
                else "🔀 Show Navigation",
                help="Hide or show the navigation panel",
                use_container_width=True,
            ):
                st.session_state.hide_navigation = not st.session_state.hide_navigation
                st.rerun()

            # Only show navigation if not hidden
            if not st.session_state.hide_navigation:
                # Add page navigation section in sidebar
                st.markdown(
                    """
                <h2 style='color: #E0E7FF; margin-top: 2rem; margin-bottom: 1rem; border-bottom: 1px solid #818CF8; padding-bottom: 0.5rem; font-weight: 600; font-size: 1.25rem;'>
                    <span style="margin-right: 8px;">📋</span> Navigation
                </h2>
                """,
                    unsafe_allow_html=True,
                )

                selected_page = st.radio(
                    "Select Page",
                    ["Dataset Overview", "Interactive Visualization"],
                    index=0 if st.session_state.current_page == "Dataset Overview" else 1,
                    help="Navigate between different views of your data",
                )

                # Update the current page in session state when changed
                if selected_page != st.session_state.current_page:
                    st.session_state.current_page = selected_page

                # View mode selection shown only when on the Interactive Visualization page
                if selected_page == "Interactive Visualization":
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
                    if st.session_state.view_mode != view_mode_map[view_mode]:
                        st.session_state.view_mode = view_mode_map[view_mode]

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
        # Create tabs for Dataset Overview and Interactive Visualization
        overview_tab, viz_tab = st.tabs(["Dataset Overview", "Interactive Visualization"])

        # Dataset Overview tab
        with overview_tab:
            st.markdown(
                """
                <div style="background-color: #f8fafc; padding: 1rem; border-radius: 0.5rem; border-left: 5px solid #3B82F6; margin-bottom: 1.5rem;">
                    <h2 style="margin-top: 0; color: #1E3A8A;">Dataset Overview</h2>
                    <p style="color: #64748b; margin-bottom: 0;">
                        Complete analysis of your dataset's structure, statistics, and quality metrics.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Use the enhanced display_df_summary function
            display_df_summary(st.session_state.data)

        # Interactive Visualization tab
        with viz_tab:
            st.markdown("## Interactive Visualization")

            # Get cached renderer
            renderer = get_pyg_renderer(st.session_state.data)

            if renderer:
                # Render selected view
                if st.session_state.view_mode == "Explorer":
                    # Use the explorer method for the standard view
                    renderer.explorer()
                elif st.session_state.view_mode == "Chart":
                    # For Chart view, we need to specify an index for which chart to display
                    # Since there might not be any charts yet, we'll safely wrap this in try/except
                    try:
                        # Show the first chart (index 0)
                        renderer.chart(0)
                    except Exception as e:
                        st.warning("⚠️ No charts available. Create a chart in Explorer view first.")
                        st.error(f"Error: {str(e)}")
                elif st.session_state.view_mode == "Data Profiling":
                    # For Data Profiling, we use explorer with "data" as the default tab
                    renderer.explorer(default_tab="data")
                elif st.session_state.view_mode == "Data Preview":
                    st.dataframe(st.session_state.data, use_container_width=True)
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
            Created by Jackson with ❤️
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
