import streamlit as st


def apply_styling():
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
        
        /* Button styling */
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
        button[data-baseweb="button"]:hover {
            background-color: rgba(255, 255, 255, 0.3) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15) !important;
        }
        
        /* Primary button styles */
        .stButton button[kind="primary"], 
        button[data-baseweb="button"][kind="primary"] {
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


def render_header():
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


def render_about_section():
    """Render the about section for the sidebar."""
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


def render_card_header(title, help_text=None):
    """Render a card header with optional help text."""
    st.markdown(
        f"""<div class="card-header">{title}</div>""",
        unsafe_allow_html=True,
        help=help_text,
    )
