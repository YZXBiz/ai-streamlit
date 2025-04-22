"""
Tests for dashboard components.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st
from dashboard.components.data_uploader import data_uploader
from dashboard.components.data_viewer import data_viewer
from dashboard.components.chat_interface import chat_interface


class TestDataUploader(unittest.TestCase):
    """Tests for the data_uploader component."""

    @patch("streamlit.file_uploader")
    @patch("streamlit.success")
    @patch("streamlit.dataframe")
    @patch("streamlit.expander")
    @patch("streamlit.subheader")
    def test_successful_csv_upload(self, mock_subheader, mock_expander, 
                                  mock_dataframe, mock_success, mock_uploader):
        """Test successful CSV file upload."""
        # Setup mock file uploader
        mock_file = MagicMock()
        mock_file.name = "test_data.csv"
        mock_uploader.return_value = mock_file
        
        # Mock context managers
        mock_expander.return_value.__enter__.return_value = MagicMock()
        
        # Create test data
        test_data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        # Mock pd.read_csv
        with patch("pandas.read_csv", return_value=test_data):
            result_df, result_filename = data_uploader()
            
            # Verify the results
            self.assertIsNotNone(result_df)
            self.assertEqual(result_filename, "test_data.csv")
            pd.testing.assert_frame_equal(result_df, test_data)
            
            # Verify the success message was shown
            mock_success.assert_called_once()
            
            # Verify dataframe preview was shown
            mock_dataframe.assert_called_once()


class TestChatInterface(unittest.TestCase):
    """Tests for the chat_interface component."""

    @patch("streamlit.chat_input")
    @patch("streamlit.chat_message")
    @patch("streamlit.markdown")
    @patch("streamlit.subheader")
    @patch("streamlit.session_state", {})
    def test_chat_message_handling(self, mock_subheader, mock_markdown, 
                                  mock_chat_message, mock_chat_input):
        """Test that messages are handled correctly."""
        # Setup mocks
        mock_chat_input.return_value = "What's the average of column A?"
        mock_context = MagicMock()
        mock_chat_message.return_value.__enter__.return_value = mock_context
        
        # Create a mock response function
        def mock_on_message(message, df):
            return f"The average of column A is {df['A'].mean()}"
        
        # Create test data
        test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        # Initialize session state
        st.session_state = {"messages": []}
        
        # Call the component
        with patch("streamlit.spinner") as mock_spinner:
            mock_spinner.return_value.__enter__.return_value = None
            chat_interface(mock_on_message, test_df)
        
        # Verify the message was added to the session state
        self.assertEqual(len(st.session_state["messages"]), 2)  # Welcome + user msg
        self.assertEqual(st.session_state["messages"][1]["role"], "user")
        self.assertEqual(st.session_state["messages"][1]["content"], 
                        "What's the average of column A?")


if __name__ == "__main__":
    unittest.main() 