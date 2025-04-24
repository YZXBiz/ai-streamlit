import os
import tempfile
from collections.abc import Sequence
from typing import Any

import streamlit as st

from chatbot.ui.helpers import table_exists


def process_file_upload(svc: Any, uploaded_files: Sequence[Any] | None) -> bool:
    """Process uploaded files and load them into DuckDB.

    Args:
        svc: DuckDB service instance
        uploaded_files: List of uploaded files from Streamlit

    Returns:
        bool: True if any new files were processed, False otherwise
    """
    if not uploaded_files:
        return False

    # Initialize session state to track processed files
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()

    # Track if any new files were processed
    any_new_files = False

    with st.spinner("Loading files into DuckDB...", show_time=True):
        for uf in uploaded_files:
            # Skip if this file was already processed
            file_id = f"{uf.name}_{uf.size}"
            if file_id in st.session_state.processed_files:
                continue

            # Get clean table name
            table_name = os.path.splitext(uf.name)[0].replace(".", "_").replace(" ", "_")

            # Check if table already exists
            if table_exists(svc, table_name):
                st.sidebar.info(f"Table '{table_name}' already exists, skipping file {uf.name}")
                st.session_state.processed_files.add(file_id)
                continue

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uf.name)[1]
            ) as tmp_file:
                tmp_file.write(uf.getbuffer())
                tmp_path = tmp_file.name

            # Use load_file_directly method from the service
            success = svc.load_file_directly(tmp_path, table_name)

            # Cleanup temp file
            os.unlink(tmp_path)

            if success:
                st.sidebar.success(f"Loaded {uf.name} into table '{table_name}'")
                # Mark file as processed
                st.session_state.processed_files.add(file_id)
                any_new_files = True
            else:
                st.sidebar.error(f"Failed to load {uf.name}")

    # Initialize tables only if new files were added
    if any_new_files:
        svc.initialize()

    return any_new_files
