"""
Session management component for the PandasAI Streamlit application.

This module provides interfaces for creating and managing chat sessions
with the FastAPI backend.
"""

import streamlit as st

from frontend.utils.auth import authenticated_request


def render_session_manager():
    """
    Render the session management interface.

    This function provides UI components for:
    - Creating new chat sessions
    - Viewing existing sessions
    - Selecting a session to work with
    """
    st.sidebar.subheader("Chat Sessions")

    # Get active sessions from backend
    response = authenticated_request("get", "/sessions")
    sessions = []

    if response and response.status_code == 200:
        sessions = response.json()

    # Create a new session
    with st.sidebar.expander("Create New Session", expanded=not sessions):
        # Get available data files
        files_response = authenticated_request("get", "")  # Endpoint for listing files

        available_files = []
        if files_response and files_response.status_code == 200:
            available_files = files_response.json()

        # Create form for new session
        with st.form("new_session_form"):
            session_name = st.text_input("Session Name")

            file_options = (
                {f"{f['name']} ({f['id']})": f["id"] for f in available_files}
                if available_files
                else {}
            )
            selected_file = st.selectbox("Data File", options=["None"] + list(file_options.keys()))

            description = st.text_area("Description (Optional)")
            submit_button = st.form_submit_button("Create Session")

            if submit_button:
                if not session_name:
                    st.error("Please provide a session name")
                else:
                    # Prepare data for API request
                    data = {"name": session_name, "description": description}

                    # Add file ID if selected
                    if selected_file != "None":
                        data["data_file_id"] = file_options[selected_file]

                    # Create session
                    create_response = authenticated_request("post", "/sessions", json=data)

                    if create_response and create_response.status_code in (200, 201):
                        st.success("Chat session created!")
                        st.rerun()
                    else:
                        error_msg = "Failed to create session"
                        if create_response:
                            error_msg = create_response.json().get("detail", error_msg)
                        st.error(error_msg)

    # Display existing sessions
    if sessions:
        st.sidebar.subheader("Your Sessions")
        for session in sessions:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(
                    f"{session['name']}",
                    key=f"session_{session['id']}",
                    help=session.get("description", ""),
                ):
                    # Set current session
                    st.session_state.current_session = session
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"delete_{session['id']}", help="Delete this session"):
                    # Confirm deletion
                    if st.sidebar.checkbox(
                        "Confirm deletion?", key=f"confirm_delete_{session['id']}"
                    ):
                        delete_response = authenticated_request(
                            "delete", f"/sessions/{session['id']}"
                        )

                        if delete_response and delete_response.status_code in (200, 204):
                            st.sidebar.success("Session deleted")

                            # Clear current session if it was the deleted one
                            if (
                                "current_session" in st.session_state
                                and st.session_state.current_session["id"] == session["id"]
                            ):
                                st.session_state.current_session = None

                            st.rerun()
                        else:
                            st.sidebar.error("Failed to delete session")
    else:
        st.sidebar.info("No sessions found. Create a new session to start chatting.")
