"""
Component for the Streamlit chat interface in the dashboard.

This module provides a reusable chat interface component for interacting with data
through natural language queries.
"""
from typing import Callable, List, Dict, Any, Optional, TypeVar, Union, cast

import pandas as pd
import streamlit as st

# Type for callback function that processes chat messages
MessageProcessor = Callable[[str, pd.DataFrame], str]


def chat_interface(
    on_message: MessageProcessor,
    df: Optional[pd.DataFrame] = None
) -> None:
    """
    Creates a chat interface component for interacting with data via chat.
    
    Parameters
    ----------
    on_message : Callable[[str, pd.DataFrame], str]
        Function to call when a message is sent. Takes the message
        and the current DataFrame as arguments, returns a response.
    df : Optional[pd.DataFrame], default=None
        The DataFrame to use for answering questions. If None, shows a message
        prompting the user to upload data.
    
    Returns
    -------
    None
        This function modifies the Streamlit UI but doesn't return any values.
    
    Notes
    -----
    This component requires Streamlit session state to store chat history.
    It initializes two session state variables if they don't exist:
    - messages: List of message objects with 'role' and 'content'
    - chat_history: Backup list for conversation history
    
    Examples
    --------
    >>> def process_message(msg: str, data: pd.DataFrame) -> str:
    ...     return f"You said: {msg}"
    >>> df = pd.DataFrame({'col1': [1, 2, 3]})
    >>> chat_interface(process_message, df)
    """
    st.subheader("Data Chat Assistant")
    
    # Initialize session state variables for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Check if we have data to query
    if df is None:
        st.info("Please upload data to chat with your data.")
        return
    
    # Display a welcome message if this is a new conversation
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            welcome_msg = (
                f"👋 Hello! I'm your data assistant. "
                f"I can help you explore the data you've uploaded "
                f"({df.shape[0]} rows × {df.shape[1]} columns). "
                f"What would you like to know about your data?"
            )
            st.markdown(welcome_msg)
            
            # Add welcome message to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg}
            )
    
    # Chat input
    if prompt := st.chat_input("Ask something about your data..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = on_message(prompt, df)
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
    
    # Add a clear button for the chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun() 