"""Clustering Dashboard Application.

A streamlined application for exploring and visualizing clustering data.
This is the main entry point for the dashboard.
"""

import streamlit as st

# Set page config at the top level
st.set_page_config(
    page_title="Clustering Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_custom_css():
    """Add custom CSS to improve the app's appearance."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 0.1rem 0.25rem rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        .metric-title {
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #0066CC;
        }
        
        .metric-trend {
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
        
        .apple-heading {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .content-container {
            background-color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 0.1rem 0.25rem rgba(0, 0, 0, 0.05);
            margin-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Display the main dashboard page."""
    # Add custom CSS
    add_custom_css()

    # Dashboard sidebar
    with st.sidebar:
        st.title("📊 Clustering Dashboard")
        st.markdown("---")

        st.markdown(
            """
            Welcome to the Clustering Dashboard! This application helps you:
            
            - Upload and explore your data
            - Visualize distributions and relationships
            - Analyze cluster assignments
            - Identify patterns in your data
            """
        )

        st.markdown("---")

        # Add information about the data source
        if "data" in st.session_state:
            st.success("✅ Data loaded")
            data_source = st.session_state.get("data_source", "unknown")
            st.info(f"Source: {data_source.capitalize()}")

            # Data shape
            df = st.session_state["data"]
            st.write(f"Rows: {df.shape[0]:,}")
            st.write(f"Columns: {df.shape[1]}")
        else:
            st.warning("No data loaded. Please upload data in the Data Upload page.")

    # Main content
    st.title("Clustering Dashboard")

    # Show getting started info on homepage
    st.markdown(
        """
        ## Getting Started
        
        Navigate through the pages in the sidebar to work with your data:
        
        1. **Data Upload**: Upload your data files or connect to Snowflake
        2. **Data Explorer**: View summary statistics and explore column distributions
        3. **Visualization**: Create interactive visualizations of your data
        
        Use the menu in the sidebar to navigate between pages.
        """
    )

    # Display feature highlights with columns
    st.subheader("Features")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### Data Exploration
            - Comprehensive data summaries
            - Statistical analysis
            - Missing value detection
            - Column profiling
            """
        )

    with col2:
        st.markdown(
            """
            ### Interactive Visualization
            - Scatter plots
            - Correlation matrices
            - Parallel coordinates
            - PyGWalker integration
            """
        )

    with col3:
        st.markdown(
            """
            ### Cluster Analysis
            - Compare clusters
            - Identify key features
            - Evaluate cluster quality
            - Visualize cluster separations
            """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
        Clustering Dashboard v0.1.0 | Made with Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
