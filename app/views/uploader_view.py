import streamlit as st


def render_uploader():
    """
    Render the file uploader component.

    Returns:
        A tuple of (uploaded_files, continue_clicked)
    """
    st.subheader("Upload Your Data")
    st.write("Upload one or more data files to analyze.")

    # Use a form to prevent auto-rerun issues
    with st.form("file_upload_form", clear_on_submit=False):
        # File uploader that accepts multiple files
        uploaded_files = st.file_uploader(
            "Choose CSV or Excel files",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            key="file_uploader",
        )

        # Display uploaded files count
        if uploaded_files:
            st.write(f"Files ready to upload: {len(uploaded_files)}")

            # Display table with file names
            file_info = [
                {"File Name": file.name, "Size (KB)": round(file.size / 1024, 2)}
                for file in uploaded_files
            ]
            st.table(file_info)

        # Continue button in form
        submit_button = st.form_submit_button(
            "Process Files", type="primary", use_container_width=True
        )

    # Only return True for continue if the form was submitted
    return uploaded_files, submit_button


def show_upload_success(file_names):
    """Show success message after files are uploaded."""
    if isinstance(file_names, list):
        files_str = ", ".join(file_names)
        st.success(f"Successfully loaded data from: {files_str}")
    else:
        st.success(f"Successfully loaded data from: {file_names}")


def show_upload_error(error_message):
    """Show error message if file upload fails."""
    st.error(f"Error: {error_message}")
