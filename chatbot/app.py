import concurrent.futures

import streamlit as st

from chatbot.ui.display import display_chat_history, display_data_schema, display_results
from chatbot.ui.file_upload import process_file_upload
from chatbot.ui.helpers import determine_complexity, enhance_query_with_context
from chatbot.ui.initialize import initialize_service
from chatbot.ui.styles import inject_styles

# Page config
st.set_page_config(
    page_title="DuckDB Natural Language Query",
    page_icon="🦆",
    layout="wide",
)

inject_styles()


# Store last query result in session state
if "last_result" not in st.session_state:
    st.session_state.last_result = None


def main() -> None:
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
