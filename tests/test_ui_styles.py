"""
Tests for UI styles module.

This module contains tests for the inject_styles function in ui/styles.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from chatbot.ui.styles import inject_styles


def test_css_style_content() -> None:
    """Test that the CSS content in the inject_styles function has expected classes."""
    # Get the function source code
    import inspect
    source = inspect.getsource(inject_styles)
    
    # Verify specific CSS classes are included
    css_classes = [
        ".main-header",
        ".section-header",
        ".stTabs [data-baseweb=\"tab-list\"]",
        ".error-box",
        ".info-box",
    ]
    
    for css_class in css_classes:
        assert css_class in source


@patch("streamlit.markdown")
def test_inject_styles_calls_markdown(mock_markdown: MagicMock) -> None:
    """Test that inject_styles calls st.markdown."""
    # Patch the markdown function to avoid actual streamlit execution
    with patch("chatbot.ui.styles.st") as mock_st:
        mock_st.markdown = mock_markdown
        
        # Call the function
        inject_styles()
        
        # Verify markdown was called once
        mock_markdown.assert_called_once()
        
        # Verify call included 'unsafe_allow_html=True'
        assert mock_markdown.call_args[1].get('unsafe_allow_html') is True 