import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding

from chatbot.config import config
from chatbot.services import EnhancedDuckDBService

# Load environment variables
load_dotenv()


@st.cache_resource
def initialize_service() -> EnhancedDuckDBService | None:
    """Initialize the DuckDB service with OpenAI embeddings.

    Returns:
        EnhancedDuckDBService or None: Initialized service instance, or None if API key is missing
    """
    # Check if OpenAI API key is available
    try:
        api_key = config.OPENAI_API_KEY.get_secret_value()
        if not api_key:
            st.error("OPENAI_API_KEY not found in configuration.")
            return None
    except ValueError:
        st.error("OPENAI_API_KEY not found or is invalid.")
        return None

    # Initialize the embedding model
    embed_model = OpenAIEmbedding(api_key=api_key)

    # Create an instance of EnhancedDuckDBService
    return EnhancedDuckDBService(
        embed_model=embed_model,
        db_path=config.DB_PATH,
        memory_type=config.MEMORY_TYPE,
        token_limit=config.MEMORY_TOKEN_LIMIT,
        chat_store_key="streamlit_user",  # Session identifier
    )
