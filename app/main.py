import streamlit as st

# App configuration must be the first Streamlit command
st.set_page_config(
    page_title="Chatbot", page_icon="💬", layout="wide", initial_sidebar_state="expanded"
)

# Only import controllers after setting page config
from app.controllers.app_controller import AppController


def main():
    """Main application entry point."""
    app = AppController()
    app.run()


if __name__ == "__main__":
    main()
