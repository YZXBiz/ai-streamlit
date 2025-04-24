"""
Chat history UI component for the Data File/Table Chatbot application.

This module provides the UI components for viewing and managing the conversation
history in the application. It allows users to view past interactions, clear the
history, and download the full conversation log.
"""

import streamlit as st

from flat_chatbot.controller import AppController


def render_history_tab(controller: AppController, container: st.container) -> None:
    """
    Render the chat history tab interface.
    
    Parameters
    ----------
    controller : AppController
        The application controller instance
    container : streamlit.container
        The Streamlit container to render content within
    """
    with container:
        st.markdown(
            "<div class='section-header'>Conversation History</div>", unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat History"):
                controller.clear_history()
                st.success("Cleared")
        with col2:
            hist = controller.svc.get_chat_history()
            if hist:
                st.download_button("📥 Download History", hist, "history.txt", mime="text/plain")
        hist = controller.svc.get_chat_history()
        if not hist:
            st.info("No history yet")
        else:
            st.text_area("History", hist, height=300, disabled=True)
