import os
import tempfile

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import Agent, DataFrame
from pandasai_openai import OpenAI

# Load environment variables
load_dotenv()


def initialize_agent(df, api_key=None):
    """
    Initialize a PandasAI agent with the given dataframe.

    Args:
        df: The pandas DataFrame to be analyzed
        api_key: OpenAI API key (optional, will use env var if not provided)

    Returns:
        A PandasAI agent configured with the dataframe
    """
    # Get API key from parameter, environment, or session
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        return None, "Missing OpenAI API Key"

    try:
        # Initialize the LLM
        llm = OpenAI(api_token=api_key)

        # Convert pandas DataFrame to PandasAI DataFrame
        pai_df = DataFrame(df)

        # Create agent with the dataframe and LLM config
        agent = Agent(
            dfs=pai_df,
            config={
                "llm": llm,
                "save_charts": False,  # Don't save charts to disk
                "verbose": True,
                "return_intermediate_steps": False,
            },
        )

        return agent, None
    except Exception as e:
        return None, str(e)


def load_dataframe(uploaded_file):
    """
    Load a dataframe from an uploaded file.

    Args:
        uploaded_file: The uploaded file object from streamlit

    Returns:
        A pandas DataFrame and any error message
    """
    if not uploaded_file:
        return None, "No file uploaded"

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Read the file based on its extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext == ".csv":
            df = pd.read_csv(tmp_path)
        elif file_ext in [".xlsx", ".xls"]:
            df = pd.read_excel(tmp_path)
        else:
            return None, "Unsupported file format"

        # Clean up the temporary file
        os.unlink(tmp_path)

        return df, None
    except Exception as e:
        return None, str(e)


def process_response(response):
    """
    Process the response from PandasAI and determine its type.

    Args:
        response: The response from PandasAI

    Returns:
        A tuple of (response_type, content)
    """
    # Check if response is a DataFrame
    if hasattr(response, "shape") and hasattr(response, "iloc"):
        return "dataframe", response

    # Check if response is a matplotlib figure or has a figure attribute
    elif hasattr(response, "figure") or str(type(response)).find("Figure") != -1:
        return "figure", response

    # Check if response is a plot/chart path (fallback for compatibility)
    elif isinstance(response, str) and (
        response.startswith("data/charts/")
        or response.startswith("exports/charts/")
        or response.endswith((".png", ".jpg", ".jpeg"))
    ):
        try:
            # Try to load the image
            import matplotlib.pyplot as plt
            from PIL import Image

            # Create a matplotlib figure with the image
            fig, ax = plt.subplots(figsize=(10, 6))

            # Check if file exists
            if os.path.exists(response):
                img = Image.open(response)
                ax.imshow(img)
                ax.axis("off")  # Hide axes

                # Return as figure for in-chat display
                return "figure", fig
            else:
                # If image file doesn't exist, return as text
                return "text", f"Chart file not found: {response}"
        except Exception:
            # Fall back to returning the image path if conversion fails
            return "image", response

    # Default to text
    else:
        return "text", response


def display_data_info(df):
    """
    Display information about the dataframe.

    Args:
        df: The pandas DataFrame to display info for
    """
    # Display data preview
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Display data shape
    st.subheader("Data Shape")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # Display column info
    st.subheader("Column Information")
    col_info = pd.DataFrame(
        {
            "Column": df.columns,
            "Type": df.dtypes,
            "Non-Null Count": df.count(),
            "Null Count": df.isna().sum(),
        }
    )
    st.dataframe(col_info)

    # Display basic statistics if numeric columns exist
    numerics = df.select_dtypes(include=["number"])
    if not numerics.empty:
        st.subheader("Numerical Statistics")
        st.dataframe(numerics.describe())
