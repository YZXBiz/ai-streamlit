"""
CSS styles for the Data File/Table Chatbot Streamlit application.

This module defines the CSS styling used throughout the application to create
a consistent and visually appealing user interface. It includes styling for
headers, tabs, status boxes, and hides various Streamlit UI elements for a 
cleaner experience.
"""

CSS = """
<style>
    .main-header {
        font-size: 2.5rem !important;
        margin-bottom: 1rem !important;
    }
    .section-header {
        font-size: 1.5rem !important;
        margin-top: 1rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #f0f0f0;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .error-box {
        background-color: #ffebe6;
        border-left: 4px solid #ff5630;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .info-box {
        background-color: #e6fcff;
        border-left: 4px solid #00b8d9;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    /* Hide Deploy button, Main Menu, and footer */
    .stDeployButton {
        visibility: hidden;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    /* Hide Running/Stop buttons */
    [data-testid="stStatusWidget"] {
        visibility: hidden;
    }
</style>
"""
