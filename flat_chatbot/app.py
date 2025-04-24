"""
Streamlit application entry point for the Data File/Table Chatbot.

This module defines the main UI layout using Streamlit components, sets up the page
configuration, and initializes the core application controller. It organizes the
interface into tabs for queries, schema browsing, and chat history.
"""

import streamlit as st

from flat_chatbot.controller import AppController
from flat_chatbot.ui.history import render_history_tab
from flat_chatbot.ui.query import render_query_tab
from flat_chatbot.ui.schema import render_schema_tab
from flat_chatbot.ui.styles import CSS
from flat_chatbot.ui.upload import render_upload_sidebar

st.set_page_config(
    page_title="Data File/Table Chatbot",
    page_icon="🤖",
    layout="centered",
)
st.markdown(CSS, unsafe_allow_html=True)

# Add main title
st.markdown("<h1 class='main-header'>🤖 Data File/Table Chatbot</h1>", unsafe_allow_html=True)

# ---- Store controller in session state to maintain state between reruns ----
if "controller" not in st.session_state:
    st.session_state.controller = AppController()
controller = st.session_state.controller

# Sidebar: data management
with st.sidebar:
    render_upload_sidebar(controller)
    # Force a rerun after upload so UI sees new tables
    if st.session_state.get("just_uploaded"):
        st.session_state.just_uploaded = False
        st.rerun()

# Gate UI based on data availability
tables = controller.get_table_list()
if not tables:
    st.warning("Please upload at least one file to begin.")
    st.markdown("""
    ### How to use this app:
    1. Upload a CSV or Parquet file using the sidebar
    2. Once uploaded, you'll be able to ask questions about your data
    3. View table schema and query history in the other tabs
    """)
else:
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Ask Questions", "📊 Tables & Schema", "📜 History"])
    render_query_tab(controller, tab1)
    render_schema_tab(controller, tab2)
    render_history_tab(controller, tab3)
