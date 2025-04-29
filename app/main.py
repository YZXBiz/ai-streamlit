import os
import streamlit as st
from dotenv import load_dotenv

# Import components and utilities
from app.components import chat_interface, reset_chat, file_uploader
from app.utils.auth_utils import login_form, logout

# Load environment variables
load_dotenv()

# App configuration
st.set_page_config(
    page_title="PandasAI Chat",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "df" not in st.session_state:
    st.session_state.df = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")

# Reset session
def reset_session():
    """Reset all session state variables and restart the session."""
    if st.button("Start New Session"):
        # Clear session state
        st.session_state.agent = None
        st.session_state.df = None
        st.session_state.chat_history = []
        st.session_state.file_name = None
        st.rerun()

# Main app
def main():
    """Main application entry point."""
    # Title and description
    st.title("🐼 PandasAI Chat")
    st.markdown("Upload your data and chat with it using natural language")
    
    # Login if not authenticated
    if not st.session_state.authenticated:
        login_form()
        return
    
    # Sidebar
    with st.sidebar:
        st.title("Options")
        
        # Logout button
        if st.button("Logout"):
            logout()
            st.rerun()
        
        # Reset session option
        reset_session()
        
        # Reset chat history
        reset_chat()
        
        # About section
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info(
            """
            This app allows you to chat with your data using natural language.
            Upload a CSV or Excel file and ask questions about your data.
            
            Powered by [PandasAI](https://github.com/sinaptik-ai/pandas-ai) and Streamlit.
            """
        )
    
    # Main area
    if st.session_state.agent is None:
        # File upload if no agent initialized
        file_uploader()
    else:
        # Chat interface if agent is initialized
        chat_interface()

if __name__ == "__main__":
    main() 