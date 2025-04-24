"""
Streamlit application for DuckDB with natural language query capabilities.

This app provides a streamlit interface for the EnhancedDuckDBService, allowing
users to upload data files, ask natural language questions, and view/download results.
"""

import concurrent.futures
import io
import os
import tempfile
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import the DuckDB service from flat structure
from duckdb_service import EnhancedDuckDBService
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="DuckDB Natural Language Query",
    page_icon="🦆",
    layout="wide",
)

# Store last query result in session state
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Custom CSS for better organization
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    .section-header {
        font-size: 1.5rem !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #f0f0f0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .error-box {
        background-color: #ffebe6;
        border-left: 4px solid #ff5630;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .info-box {
        background-color: #e6fcff;
        border-left: 4px solid #00b8d9;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    /* Hide Deploy button, Main Menu, and footer */
    .stDeployButton {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    /* Hide Running/Stop buttons */
    [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def initialize_service():
    """Initialize the DuckDB service with OpenAI embeddings.

    Returns:
        EnhancedDuckDBService: Initialized service instance
    """
    # Use OPENAI_API_KEY directly from settings
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found in settings.")
        return None

    # Initialize the embedding model
    embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

    # Create an instance of EnhancedDuckDBService
    return EnhancedDuckDBService(
        embed_model=embed_model,
        memory_type="summary",  # Use summary memory for better context retention
        token_limit=4000,  # Adjust token limit as needed
        chat_store_key="streamlit_user",  # Session identifier
    )


def display_results(result: dict[str, Any]) -> None:
    """Display query results in the Streamlit UI.

    Args:
        result: Output dictionary from process_query
    """
    # Store the result in session state for later use
    st.session_state.last_result = result

    if not result["success"]:
        st.markdown(
            f"""<div class="error-box">
                <strong>Error:</strong> {result.get("error", "Unknown error occurred")}
                </div>""",
            unsafe_allow_html=True,
        )
        return

    # Display the response
    st.markdown("<div class='section-header'>Response</div>", unsafe_allow_html=True)
    st.write(result["data"])

    # Add clean download options if raw data is available
    if "raw_data" in result:
        try:
            df = pd.DataFrame(result["raw_data"])

            # Display the raw data as a DataFrame with a clear header
            st.markdown("<div class='section-header'>Data Table View</div>", unsafe_allow_html=True)
            if df.empty:
                st.warning("No data to display")
            elif df.shape[0] > 10:
                st.info("Output size is too large to display, showing first 10 rows")
                st.dataframe(df.head(10), use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)

            # Show table info
            num_rows, num_cols = df.shape
            st.info(f"Table contains {num_rows} rows and {num_cols} columns")

            # Clean divider
            st.markdown("---")

            # Simple download section
            st.markdown("#### Download Options")

            # Create three columns for the buttons
            col1, col2, col3 = st.columns(3)

            # Column 1: CSV Download
            with col1:
                csv_bytes = df.to_csv(index=False).encode()
                st.download_button(
                    "💾 Download as CSV",
                    data=csv_bytes,
                    file_name="query_results.csv",
                    mime="text/csv",
                    key="simple_csv_btn",
                    use_container_width=True,
                )

            # Column 2: Excel Download
            with col2:
                try:
                    buffer = io.BytesIO()
                    df.to_excel(buffer, index=False, engine="openpyxl")
                    buffer.seek(0)
                    st.download_button(
                        "📊 Download as Excel",
                        data=buffer,
                        file_name="query_results.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="simple_excel_btn",
                        use_container_width=True,
                    )
                except Exception:
                    st.button(
                        "📊 Download as Excel (Unavailable)",
                        disabled=True,
                        key="disabled_excel_btn",
                        use_container_width=True,
                    )

            # Column 3: Save to Snowflake (UI only)
            with col3:
                st.button(
                    "❄️ Save to Snowflake",
                    key="snowflake_btn",
                    use_container_width=True,
                    help="Save this dataset to a Snowflake table (Coming soon)",
                )
        except Exception as e:
            st.warning(f"Could not convert result to DataFrame: {str(e)}")
            return

    # Show SQL query in an expander
    if "sql_query" in result:
        with st.expander("Show Generated SQL Query", expanded=False):
            st.code(result.get("sql_query", ""), language="sql")


def determine_complexity(tables: list[str]) -> str:
    """Automatically determine query complexity based on number of tables.

    Args:
        tables: List of table names available in the database

    Returns:
        str: "advanced" if 2 or more tables exist, "simple" otherwise
    """
    return "advanced" if len(tables) >= 2 else "simple"


def display_data_schema(svc):
    """Display database schema information.

    Args:
        svc: DuckDB service instance with table information
    """
    if not hasattr(svc, "tables") or not svc.tables:
        st.info("No tables loaded. Please upload data files to view schema information.")
        return

    st.markdown("<div class='section-header'>Database Schema</div>", unsafe_allow_html=True)

    # Get schema details for all tables
    schema_info = svc.get_schema_info()

    # Display each table's schema in an expander
    for table in svc.tables:
        with st.expander(f"Table: {table}", expanded=False):
            # Show table preview
            try:
                df = svc.execute_query(f"SELECT * FROM {table} LIMIT 5")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error previewing table: {str(e)}")

            # Show columns and types
            if table in schema_info["columns"]:
                st.markdown("##### Columns")
                columns_df = svc.execute_query(f"DESCRIBE {table}")
                st.dataframe(columns_df[["column_name", "column_type"]], use_container_width=True)


def display_chat_history(svc):
    """Display the chat history.

    Args:
        svc: DuckDB service instance with chat memory
    """
    st.markdown("<div class='section-header'>Conversation History</div>", unsafe_allow_html=True)

    # Display action buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat History", key="clear_history_btn", use_container_width=True):
            svc.clear_chat_history()
            st.success("Chat history cleared")
            st.rerun()

    # Get chat history
    chat_history = svc.get_chat_history()

    with col2:
        if chat_history:
            # Create download button for the chat history
            st.download_button(
                "Download Chat History",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain",
                key="download_history_btn",
                use_container_width=True,
            )

    # Display formatted chat history
    if not chat_history:
        st.info("No conversation history yet. Start asking questions!")
        return

    # Display formatted chat history
    st.text_area("Conversation", value=chat_history, height=400, disabled=True)


def table_exists(svc, table_name: str) -> bool:
    """Check if a table already exists in the database.

    Args:
        svc: DuckDB service instance
        table_name: Name of the table to check

    Returns:
        bool: True if the table exists, False otherwise
    """
    try:
        # Try to execute a simple query against the table
        svc.execute_query(f"SELECT 1 FROM {table_name} LIMIT 0")
        return True
    except Exception:
        return False


def process_file_upload(svc, uploaded_files):
    """Process uploaded files and load them into DuckDB.

    Args:
        svc: DuckDB service instance
        uploaded_files: List of uploaded files from Streamlit

    Returns:
        bool: True if any new files were processed, False otherwise
    """
    if not uploaded_files:
        return False

    # Initialize session state to track processed files
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Track if any new files were processed
    any_new_files = False

    with st.spinner("Loading files into DuckDB...", show_time=True):
        for uf in uploaded_files:
            # Skip if this file was already processed
            file_id = f"{uf.name}_{uf.size}"
            if file_id in st.session_state.processed_files:
                continue

            # Get clean table name
            table_name = os.path.splitext(uf.name)[0].replace(".", "_").replace(" ", "_")

            # Check if table already exists
            if table_exists(svc, table_name):
                st.sidebar.info(f"Table '{table_name}' already exists, skipping file {uf.name}")
                st.session_state.processed_files.add(file_id)
                continue

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uf.name)[1]
            ) as tmp_file:
                tmp_file.write(uf.getbuffer())
                tmp_path = tmp_file.name

            # Use load_file_directly method from the service
            success = svc.load_file_directly(tmp_path, table_name)

            # Cleanup temp file
            os.unlink(tmp_path)

            if success:
                st.sidebar.success(f"Loaded {uf.name} into table '{table_name}'")
                # Mark file as processed
                st.session_state.processed_files.add(file_id)
                any_new_files = True
            else:
                st.sidebar.error(f"Failed to load {uf.name}")

    # Initialize tables only if new files were added
    if any_new_files:
        svc.initialize()

    return any_new_files


def enhance_query_with_context(query: str, tables: list[str]) -> str:
    """Enhance a natural language query with explicit table context.

    Args:
        query: Original user query
        tables: List of available table names

    Returns:
        str: Enhanced query with table context
    """
    table_list = ", ".join(tables)
    table_context = f"Available tables: {table_list}. Only use these tables in your query."

    # Add a note about not using non-existent tables like conversation_history
    note = "Important: Do not reference tables that are not in the above list. Tables like 'conversation_history' do not exist."

    enhanced_query = f"{table_context}\n{note}\n\nQuestion: {query}"
    return enhanced_query


def main():
    """Main application function."""
    # Application title
    st.markdown(
        "<h1 class='main-header'>🦆 DuckDB Natural Language Query</h1>", unsafe_allow_html=True
    )

    # Initialize the service
    if "duckdb_service" not in st.session_state:
        st.session_state.duckdb_service = initialize_service()

    svc = st.session_state.duckdb_service
    if svc is None:
        st.error("Failed to initialize DuckDB service. Please check your OpenAI API key.")
        st.stop()

    # Sidebar for data management
    with st.sidebar:
        st.markdown("<div class='section-header'>Data Management</div>", unsafe_allow_html=True)

        # File uploader section
        with st.expander("Upload Files", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload CSV/Parquet files",
                type=["csv", "parquet"],
                accept_multiple_files=True,
                key="uploader",
            )

        # Data operations section
        st.markdown("##### Data Operations")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear All Data", use_container_width=True):
                svc.clear_data()
                # Clear processed files tracking
                if "processed_files" in st.session_state:
                    st.session_state.processed_files = set()
                st.success("All data cleared")
                st.rerun()

        with col2:
            if st.button("Refresh Tables", use_container_width=True):
                svc.initialize()
                st.success("Tables refreshed")

        # Table summary
        if hasattr(svc, "tables") and svc.tables:
            st.markdown("##### Available Tables")
            for i, table in enumerate(svc.tables):
                st.markdown(f"{i + 1}. **{table}**")

            # Show complexity mode
            complexity_mode = determine_complexity(svc.tables)
            mode_icon = "✨" if complexity_mode == "advanced" else "🔍"
            st.info(f"{mode_icon} Using **{complexity_mode}** query mode")
        else:
            st.warning("No tables available")

    # Process file uploads - track if we need to rerun the app
    new_files_added = process_file_upload(svc, uploaded_files)

    # Rerun only if new files were added to refresh the UI
    if new_files_added:
        st.rerun()

    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📊 Tables & Schema", "📜 History"])

    # Tab 1: Query Interface
    with tab1:
        if not hasattr(svc, "tables") or not svc.tables:
            st.warning("Please upload data files using the sidebar to begin querying")
        else:
            st.markdown(
                "<div class='section-header'>Natural Language Query</div>", unsafe_allow_html=True
            )

            # Display available tables info to help users
            available_tables = ", ".join(svc.tables)
            st.markdown(
                f"""<div class="info-box">
                    <strong>Available Tables:</strong> {available_tables}
                    <br>You can ask questions about these tables or request joins between them.
                    </div>""",
                unsafe_allow_html=True,
            )

            # Use a form to capture Enter key press
            with st.form(key="query_form"):
                # Question input
                query = st.text_input(
                    "Ask your question about the data...",
                    key="query",
                    placeholder="e.g., What is the average age of users?",
                )

                # Submit button (will be triggered by Enter key as well)
                col1, col2, col3 = st.columns([2, 3, 2])
                with col2:
                    submit_button = st.form_submit_button(
                        "📤 Send Question", use_container_width=True, type="primary"
                    )

            # Early return if form not submitted or query is empty
            # This keeps the UI stable without proceeding to query processing
            if not submit_button or not query:
                return
            
            # Automatically determine complexity based on number of tables
            complexity = determine_complexity(svc.tables)

            # Enhance query with explicit table context
            enhanced_query = enhance_query_with_context(query, svc.tables)

            with st.spinner("Processing your question...", show_time=True):
                try:
                    # Set a timeout of 20 seconds for query processing
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        # Submit the query processing task to the executor
                        future = executor.submit(
                            svc.process_query, enhanced_query, "natural_language", complexity
                        )

                        try:
                            # Wait for the result with a timeout
                            result = future.result(timeout=20)
                            display_results(result)
                        except concurrent.futures.TimeoutError:
                            st.error(
                                "Query processing timed out after 20 seconds. Please try a simpler query or check your data."
                            )
                except Exception as e:
                    error_msg = str(e)
                    if "table" in error_msg.lower() and "does not exist" in error_msg.lower():
                        # Special handling for table not found errors
                        st.markdown(
                            f"""<div class="error-box">
                                <strong>Table Error:</strong> {error_msg}
                                <br><br>This may happen if the AI attempts to use a table that doesn't exist.
                                <br>Available tables in your database are: <strong>{available_tables}</strong>
                                <br><br>Try rephrasing your question to specifically mention these table(s).
                                </div>""",
                            unsafe_allow_html=True,
                        )
                    else:
                        # General error handling
                        st.error(f"An error occurred: {error_msg}")

    # Tab 2: Tables & Schema Information
    with tab2:
        display_data_schema(svc)

    # Tab 3: Conversation History
    with tab3:
        display_chat_history(svc)


if __name__ == "__main__":
    main()
