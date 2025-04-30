import streamlit as st

from app.views.layout_view import render_about_section, render_card_header


def render_sidebar():
    """
    Render the sidebar with all components.

    Returns:
        Dictionary of action buttons that were clicked
    """
    actions = {"logout": False, "new_chat": False, "clear_chat": False}

    with st.sidebar:
        # User actions section
        render_card_header("🔐 User Actions", help_text="Actions for managing your session")

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
                actions["logout"] = True

        # New chat button
        with col2:
            if st.button(
                "🔄 New Chat",
                use_container_width=True,
                key="new_chat_btn",
                help="Start completely fresh with a new dataset and conversation",
            ):
                actions["new_chat"] = True

        st.markdown("<br>", unsafe_allow_html=True)

        # Chat controls section
        render_card_header(
            "💬 Chat Options", help_text="Options for managing your chat conversation"
        )

        # Reset chat button
        if st.button(
            "Clear Chat",
            key="clear_chat_btn",
            help="Clear only the conversation history while keeping the current dataset",
        ):
            actions["clear_chat"] = True

        st.markdown("<br>", unsafe_allow_html=True)

        # About section
        render_about_section()

    return actions
