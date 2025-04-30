import os

import streamlit as st
from dotenv import load_dotenv
from streamlit_cookies_manager import EncryptedCookieManager

# Import components and utilities
from app.components import chat_interface, file_uploader, reset_chat
from app.utils.auth_utils import auth_manager, session_manager

# -----------------------------------------------------------------------------
# CONFIGURATION & ENVIRONMENT
# -----------------------------------------------------------------------------

# App configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Chatbot", page_icon="💬", layout="wide", initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()


# -----------------------------------------------------------------------------
# UI STYLING
# -----------------------------------------------------------------------------


def load_app_styling():
    """Apply custom styling to the application UI."""
    st.markdown(
        """
    <style>
        /* Import Roboto font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
        
        /* Apply Roboto to all content */
        html, body, [class*="css"] {
            font-family: 'Roboto', sans-serif !important;
        }
        
        /* Rich sidebar background */
        [data-testid="stSidebarContent"] {
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
            color: white;
            padding: 1.5rem 1rem;
            font-family: 'Roboto', sans-serif;
        }
        
        /* Stylish section cards */
        .sidebar-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(5px);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Card headers */
        .card-header {
            font-size: 16px;
            font-weight: 600;
            color: white;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 8px;
        }
        
        /* Comprehensive button styling for all scenarios */
        /* Base button styles */
        .stButton button, 
        .stButton > button, 
        button[data-baseweb="button"],
        div.stButton > button[data-baseweb],
        div:has(> button[data-baseweb="button"]) button {
            border-radius: 8px !important;
            font-weight: 500 !important;
            border: none !important;
            padding: 10px 16px !important;
            transition: all 0.3s ease !important;
            font-size: 14px !important;
            background-color: rgba(255, 255, 255, 0.2) !important;
            color: white !important;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Hover styles */
        .stButton button:hover, 
        .stButton > button:hover, 
        button[data-baseweb="button"]:hover,
        div.stButton > button[data-baseweb]:hover,
        div:has(> button[data-baseweb="button"]) button:hover {
            background-color: rgba(255, 255, 255, 0.3) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Primary button styles */
        .stButton button[kind="primary"], 
        .stButton > button[kind="primary"], 
        button[data-baseweb="button"][kind="primary"],
        div.stButton > button[data-baseweb][kind="primary"],
        div:has(> button[data-baseweb="button"][kind="primary"]) button,
        [data-testid="stButton"] [kind="primary"] button {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5) !important;
            color: white !important;
        }
        
        /* Primary button hover */
        .stButton button[kind="primary"]:hover, 
        .stButton > button[kind="primary"]:hover, 
        button[data-baseweb="button"][kind="primary"]:hover,
        div.stButton > button[data-baseweb][kind="primary"]:hover,
        div:has(> button[data-baseweb="button"][kind="primary"]) button:hover,
        [data-testid="stButton"] [kind="primary"] button:hover {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5) !important;
            box-shadow: 0 4px 12px rgba(0, 210, 255, 0.4) !important;
        }
        
        /* Fix for Logout button specifically */
        button:has(span:contains("Logout")),
        button:has(span:contains("🚪")) {
            background: linear-gradient(90deg, #00d2ff, #3a7bd5) !important;
            color: white !important;
        }
        
        /* About section */
        .about-box {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        
        .about-box h4 {
            margin: 0 0 10px 0;
            color: white;
            font-size: 15px;
            font-weight: 600;
        }
        
        .about-box p {
            color: rgba(255, 255, 255, 0.8);
            font-size: 13px;
            line-height: 1.5;
            margin: 0 0 10px 0;
        }
        
        .about-box .footer {
            color: rgba(255, 255, 255, 0.5);
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# SESSION & COOKIE MANAGEMENT
# -----------------------------------------------------------------------------


def initialize_session_state():
    """Initialize all session state variables with default values."""
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


def setup_cookie_manager():
    """Initialize and configure the encrypted cookie manager."""
    # Create cookie manager with security key from environment
    cookie_manager = EncryptedCookieManager(
        prefix="chatbot",
        password=os.getenv("COOKIE_SECRET", "fallback-secret-key"),
    )

    # Check for existing authentication in cookies
    if cookie_manager.ready():
        if cookie_manager.get("user_authenticated") == "true":
            st.session_state.authenticated = True
        else:
            st.session_state.authenticated = False

    return cookie_manager


# -----------------------------------------------------------------------------
# UI COMPONENTS
# -----------------------------------------------------------------------------


def render_app_header():
    """Render the application header and title section."""
    st.markdown(
        """
    <div style="text-align: center; padding: 1rem; margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">💬 Chatbot</h1>
        <p style="font-size: 1.2rem; color: #4e4376; font-weight: 300;">Upload your data and chat with it using natural language</p>
        <div style="height: 4px; width: 100px; background: linear-gradient(90deg, #00d2ff, #3a7bd5); margin: 1rem auto;"></div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar(cookies):
    """Render the sidebar with all its components."""
    with st.sidebar:
        # User actions section
        st.markdown(
            """<div class="card-header">🔐 User Actions</div>""",
            unsafe_allow_html=True,
            help="Actions for managing your session",
        )

        # Action buttons in two columns
        col1, col2 = st.columns(2)

        # Logout button
        with col1:
            if st.button(
                "🚪 Logout",
                use_container_width=True,
                type="primary",
                key="logout_btn",
                help="Log out of your current session",
            ):
                handle_logout(cookies)

        # New chat button
        with col2:
            if st.button(
                "🔄 New Chat",
                use_container_width=True,
                key="new_chat_btn",
                help="Start completely fresh with a new dataset and conversation",
            ):
                session_manager.reset_session()
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat controls section
        st.markdown(
            """<div class="card-header">💬 Chat Options</div>""",
            unsafe_allow_html=True,
            help="Options for managing your chat conversation",
        )

        # Reset chat button
        reset_chat()

        st.markdown("<br>", unsafe_allow_html=True)

        # About section in a card
        render_about_section()


def render_about_section():
    """Render the about section in the sidebar."""
    st.markdown(
        """
    <div class="sidebar-card">
        <div class="card-header">ℹ️ About</div>
        <div class="about-box">
            <h4>Data Chat Assistant</h4>
            <p>Upload your data and ask questions using natural language.</p>
            <div class="footer">Created by Jackson Yang with ❤️</div>
        </div>
    </div>
        """,
        unsafe_allow_html=True,
    )


def handle_logout(cookies):
    """Handle the logout process."""
    # Clear cookie on logout
    if cookies.ready():
        cookies["user_authenticated"] = ""
        cookies.save()

    # Clear session state
    session_manager.logout()

    # Refresh the page
    st.rerun()


def handle_login(cookies):
    """Handle the login process."""
    if auth_manager.login_form():
        # Set cookie on successful login - expires in 7 days
        cookies["user_authenticated"] = "true"

        # Force an immediate save
        cookies.save()

        # Update session state
        st.session_state.authenticated = True

        # Refresh the page
        st.rerun()

    return False


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------


def main():
    """Main application entry point."""
    # Apply custom styling
    load_app_styling()

    # Set up session state and cookies
    initialize_session_state()
    cookies = setup_cookie_manager()

    # Render the app header
    render_app_header()

    # Authentication check
    if not st.session_state.authenticated:
        handle_login(cookies)
        return

    # Render sidebar components
    render_sidebar(cookies)

    # Render main content area based on application state
    if st.session_state.agent is None:
        # Show file upload interface if no agent is initialized
        file_uploader()
    else:
        # Show chat interface if agent is initialized
        chat_interface()


if __name__ == "__main__":
    main()
