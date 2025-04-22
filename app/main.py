"""
Main entry point for the Chatbot Web Application.

This module initializes the Streamlit application and sets up the main UI components.
"""
import os
import sys
from pathlib import Path
import uuid
import streamlit as st
from dotenv import load_dotenv

# Import components
from app.components.data_upload import render_data_upload
from app.components.visualization import render_visualization
from app.components.chat_interface import render_chat_interface

# Import services
from app.services.data_service import DataService
from app.services.query_service import QueryService
from app.services.persistence import PersistenceService

# Import utils for logging and error handling
from app.utils import (
    configure_logging,
    get_logger,
    mdc_context,
    AppError,
    InvalidConfigurationError,
)

# Load environment variables
load_dotenv()

# Configure logging
logs_dir = os.environ.get("LOGS_DIR", os.path.join(os.getcwd(), "logs"))
log_level = os.environ.get("LOG_LEVEL", "INFO")

# Create logs directory if it doesn't exist
Path(logs_dir).mkdir(parents=True, exist_ok=True)

# Configure application-wide logging
configure_logging(
    log_level=log_level,
    log_file=os.path.join(logs_dir, "app.log"),
    console=True,
)

# Get logger for this module
logger = get_logger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Data Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

def initialize_session_state():
    """Initialize the session state variables if they don't exist."""
    if "data" not in st.session_state:
        st.session_state.data = None
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "visualization_state" not in st.session_state:
        st.session_state.visualization_state = {
            "last_chart_type": None,
            "filters": {},
            "aggregations": []
        }
    
    if "session_id" not in st.session_state:
        # Generate a unique session ID
        st.session_state.session_id = str(uuid.uuid4())
        logger.info(f"New session started with ID: {st.session_state.session_id}")


def check_environment():
    """Check that required environment variables are set."""
    required_vars = []
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        missing_vars_str = ", ".join(missing_vars)
        error_msg = f"Missing required environment variables: {missing_vars_str}"
        logger.error(error_msg)
        raise InvalidConfigurationError(error_msg)


def main():
    """Main application function."""
    try:
        # Check environment variables
        check_environment()
        
        # Initialize session state
        initialize_session_state()
        
        # Add session context to logs
        with mdc_context(session_id=st.session_state.session_id):
            logger.info("Application started")
            
            # Header
            st.title("Data Chat Assistant")
            st.markdown(
                """
                Upload data, visualize it, and ask questions in natural language!
                """
            )
            
            # Sidebar for data upload
            with st.sidebar:
                st.header("Data Source")
                render_data_upload()
            
            # Main content area - tabs for visualization and chat
            tab1, tab2 = st.tabs(["Data Visualization", "Chat"])
            
            with tab1:
                if st.session_state.data is not None:
                    render_visualization(st.session_state.data)
                else:
                    st.info("Please upload data or connect to Snowflake to visualize data.")
            
            with tab2:
                if st.session_state.data is not None:
                    render_chat_interface()
                else:
                    st.info("Please upload data or connect to Snowflake to start chatting about your data.")
            
            # Footer
            st.markdown("---")
            st.markdown("© 2023 Data Chat Assistant")
            
    except InvalidConfigurationError as e:
        st.error(f"Configuration Error: {e.message}")
        logger.error(f"Application failed to start due to configuration error: {e.message}")
        
    except AppError as e:
        st.error(f"Application Error: {e.message}")
        logger.error(f"Application error: {e.code} - {e.message}")
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        logger.exception(f"Unhandled exception in main: {str(e)}")


if __name__ == "__main__":
    main() 