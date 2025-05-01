import streamlit as st

from app.views.layout_view import render_card_header


def render_output_type_selector():
    """
    Render a simple selector for choosing the output format.

    Returns:
        str: The selected output type, or None for auto
    """
    with st.sidebar:
        # Add a small space
        st.markdown("<br>", unsafe_allow_html=True)

        # Use the same card header style as other sidebar components
        render_card_header("🔄 Response Format", help_text="Control how answers are presented")

        # Use a select box instead of radio buttons
        output_type = st.selectbox(
            "",  # Remove the label since we have a card header
            ["Auto", "String", "Dataframe", "Plot"],
            index=0,
        )

        # Map UI-friendly names to PandasAI parameter values
        if output_type == "Auto":
            return None
        elif output_type == "String":
            return "string"
        elif output_type == "Dataframe":
            return "dataframe"
        elif output_type == "Chart":
            return "chart"
