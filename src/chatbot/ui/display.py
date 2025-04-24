import io
from typing import Any

import pandas as pd
import streamlit as st


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


def display_data_schema(svc: Any) -> None:
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


def display_chat_history(svc: Any) -> None:
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
