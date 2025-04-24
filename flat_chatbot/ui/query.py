"""
Query UI component for the Data File/Table Chatbot application.

This module provides the UI components for the natural language query tab
in the application. It handles displaying the query form, processing queries,
and visualizing the results including data tables and download options.
"""

import io
import streamlit as st
import pandas as pd

def render_query_tab(controller, container):
    """
    Renders the natural language query tab in the Streamlit UI.
    
    This tab allows users to ask questions in natural language about their data
    and view the results. The function handles fetching tables, displaying
    the query form, and processing query results.
    
    Args:
        controller: The application controller instance
        container: The Streamlit container to render the UI in
    """
    with container:
        tbls = controller.get_table_list()
        if not tbls:
            st.warning("Upload data first")
            return

        st.markdown("<div class='section-header'>Natural Language Query</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='info-box'><strong>Available Tables:</strong> {', '.join(tbls)}</div>", unsafe_allow_html=True)

        with st.form("query_form"):
            q = st.text_input("Ask your question...")
            submitted = st.form_submit_button("📤 Send Question")
        if not submitted or not q:
            return

        mode = "advanced" if len(tbls)>=2 else "simple"
        prompt = f"Available tables: {', '.join(tbls)}\n\nQuestion: {q}"
        with st.spinner("Processing..."):
            res = controller.ask(prompt, mode)
        _display_results(res)

def _display_results(result):
    """
    Display query results in the UI including data tables and download options.
    
    This function handles the visualization of query responses, including
    text responses, data tables, and providing download options for the data.
    
    Args:
        result: Dict containing the query results with keys for success,
               data, raw_data, and potentially sql_query and error messages
    """
    controller = None  # not needed here
    st.session_state.last_result = result
    if not result["success"]:
        st.error(result.get("error","Unknown"))
        return

    st.markdown("<div class='section-header'>Response</div>", unsafe_allow_html=True)
    st.write(result["data"])

    raw = result.get("raw_data")
    if raw is not None:
        df = pd.DataFrame(raw)
        st.markdown("<div class='section-header'>Data Table View</div>", unsafe_allow_html=True)
        if df.empty:
            st.warning("No data")
        else:
            st.dataframe(df.head(10) if len(df)>10 else df, use_container_width=True)
        st.info(f"{len(df)} rows × {len(df.columns)} cols")
        st.markdown("---")
        st.markdown("#### Download Options")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.download_button("💾 CSV", df.to_csv(index=False), "results.csv", mime="text/csv")
        with c2:
            buf = io.BytesIO(); df.to_excel(buf, index=False, engine="openpyxl"); buf.seek(0)
            st.download_button("📊 Excel", buf, "results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c3:
            st.button("❄️ Save to Snowflake", disabled=True)

    if "sql_query" in result:
        with st.expander("Show Generated SQL Query"):
            st.code(result["sql_query"], language="sql")
