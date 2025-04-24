import os

import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

from chatbot.services import EnhancedDuckDBService

# Load environment variables
load_dotenv()


@st.cache_resource
def initialize_service() -> EnhancedDuckDBService | None:
    """Initialize the DuckDB service with OpenAI embeddings.

    Returns:
        EnhancedDuckDBService or None: Initialized service instance, or None if API key is missing
    """
    # Use OPENAI_API_KEY directly from settings
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not found in settings.")
        return None

    # Initialize the embedding model
    embed_model = OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))

    # Create an instance of EnhancedDuckDBService
    return EnhancedDuckDBService(
        embed_model=embed_model,
        memory_type="summary",  # Use summary memory for better context retention
        token_limit=4000,  # Adjust token limit as needed
        chat_store_key="streamlit_user",  # Session identifier
    )
