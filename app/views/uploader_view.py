import streamlit as st


def render_uploader():
    """
    Render the file upload interface.

    Returns:
        Tuple of (uploaded_file, continue_button_clicked)
    """
    st.subheader("Upload Data")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    continue_clicked = False

    if uploaded_file is not None:
        # Add button to continue to chat
        continue_clicked = st.button(
            "Continue to Chat", type="primary", use_container_width=True, key="continue_to_chat_btn"
        )
    else:
        st.info("Please upload a CSV or Excel file to get started.")

    return uploaded_file, continue_clicked


def show_upload_success(filename):
    """Show success message after file upload."""
    st.success(f"File '{filename}' uploaded and processed successfully!")


def show_upload_error(error_message):
    """Show error message for file upload issues."""
    st.error(f"Error: {error_message}")
