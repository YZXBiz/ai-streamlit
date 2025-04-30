import os
from unittest import mock

import pandas as pd
import pytest
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from app.controllers.data_controller import DataController


@pytest.fixture
def mock_uploaded_files():
    """Mock multiple uploaded files."""
    # Create mock uploaded files
    customer_file = mock.Mock(spec=UploadedFile)
    customer_file.name = "customers.csv"
    
    order_file = mock.Mock(spec=UploadedFile)
    order_file.name = "orders.csv"
    
    # Configure getvalue to return CSV content
    customer_file.getvalue.return_value = b"customer_id,name,email\n1,John,john@example.com\n2,Jane,jane@example.com"
    order_file.getvalue.return_value = b"order_id,customer_id,amount\n101,1,100\n102,2,200"
    
    return [customer_file, order_file]


@pytest.fixture
def setup_controller():
    """Set up a data controller for testing."""
    # Initialize session state
    if not hasattr(st, "session_state"):
        st.session_state = {}
    
    # Clear and reinitialize session state
    st.session_state.clear()
    
    # Create controller
    return DataController()


def test_handle_file_upload_multiple_files(setup_controller, mock_uploaded_files):
    """Test handling multiple file uploads with automatic table naming."""
    controller = setup_controller
    
    # Mock the render_uploader function to return our test files
    with mock.patch("app.views.uploader_view.render_uploader") as mock_render_uploader:
        mock_render_uploader.return_value = (mock_uploaded_files, True)
        
        # Mock the data model to return expected dataframes with auto-generated names
        with mock.patch.object(controller.data_model, "load_multiple_dataframes") as mock_load_dataframes:
            # Create dictionary of dataframes with auto-generated names
            dataframes = {
                "customers": pd.DataFrame({"customer_id": [1, 2], "name": ["John", "Jane"]}),
                "orders": pd.DataFrame({"order_id": [101, 102], "customer_id": [1, 2], "amount": [100, 200]})
            }
            mock_load_dataframes.return_value = (dataframes, [])
            
            # Mock the agent model to return a fake agent
            with mock.patch.object(controller.agent_model, "create_agent") as mock_create_agent:
                mock_agent = mock.MagicMock()
                mock_create_agent.return_value = (mock_agent, None)
                
                # Run the upload handler
                result = controller.handle_file_upload()
                
                # Verify it returned True (success)
                assert result is True
                
                # Check that multiple dataframes were passed to create_agent
                mock_create_agent.assert_called_once()
                args, kwargs = mock_create_agent.call_args
                
                # First arg should be a dictionary of dataframes
                assert isinstance(args[0], dict)
                assert len(args[0]) == 2
                assert "customers" in args[0]
                assert "orders" in args[0]
                
                # Check session state
                assert st.session_state.agent is mock_agent
                assert "dfs" in st.session_state
                assert len(st.session_state.dfs) == 2
                assert "customers" in st.session_state.dfs
                assert "orders" in st.session_state.dfs
                assert "table_names" in st.session_state
                assert len(st.session_state.table_names) == 2
                assert "customers" in st.session_state.table_names
                assert "orders" in st.session_state.table_names 