"""
Custom styles for the PandasAI Streamlit application.

This module provides functions for applying custom CSS styles to the Streamlit app.
"""

import streamlit as st


def apply_custom_styles():
    """
    Apply custom CSS styles to the Streamlit application.
    
    This function injects custom CSS into the Streamlit app to override default styles.
    """
    st.markdown("""
    <style>
        /* Main headers */
        .main-header {
            color: #4CAF50; 
            font-size: 2.5rem; 
            font-weight: bold;
            margin-bottom: 1rem;
        }
        
        /* Sub headers */
        .sub-header {
            color: #2E7D32; 
            font-size: 1.8rem; 
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        /* Info boxes */
        .info-box {
            background-color: #E8F5E9; 
            padding: 1rem; 
            border-radius: 0.5rem; 
            margin-bottom: 1rem;
            border-left: 5px solid #4CAF50;
        }
        
        /* Chat messages */
        .stChatMessage {
            border-radius: 10px;
        }
        
        /* User message styling */
        .user-message {
            background-color: #E3F2FD;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        /* Assistant message styling */
        .assistant-message {
            background-color: #F1F8E9;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        
        /* Code blocks in chat */
        .chat-code {
            background-color: #F5F5F5;
            padding: 10px;
            border-radius: 5px;
            font-family: monospace;
            margin-top: 5px;
            margin-bottom: 5px;
            overflow-x: auto;
        }
        
        /* Sidebar styling */
        .sidebar-header {
            color: #2E7D32;
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
        }
        
        .stButton>button:hover {
            background-color: #2E7D32;
        }
    </style>
    """, unsafe_allow_html=True)
