"""
Assortment Analyst module for analyzing retail product data.

This module provides the core business logic for interacting with retail
assortment data through natural language queries, SQL analysis, and visualization,
leveraging the EnhancedDuckDBService for database and AI capabilities.
"""

import os
import traceback
from typing import Literal

import pandas as pd
import streamlit as st
from llama_index.embeddings.openai import OpenAIEmbedding  # type: ignore

from assortment_chatbot.config.settings import SETTINGS
from assortment_chatbot.services.duckdb_service import EnhancedDuckDBService
from assortment_chatbot.utils.logging import get_logger

logger = get_logger(__name__)


class AssortmentAnalyst:
    """
    Assortment Analyst orchestrator using EnhancedDuckDBService.

    This class acts as a facade, coordinating between the UI and the
    EnhancedDuckDBService to load data, process natural language queries,
    and manage chat state.
    """

    def __init__(self) -> None:
        """Initialize the AssortmentAnalyst with EnhancedDuckDBService."""
        logger.info("Initializing AssortmentAnalyst")

        # Get settings
        self.agent_settings = SETTINGS.agent_settings
        self.duckdb_settings = SETTINGS.duckdb_settings

        # Ensure OpenAI API key is available
        if not SETTINGS.OPENAI_API_KEY:
            error_msg = (
                "OpenAI API key is not configured. "
                "Please set OPENAI_API_KEY in your environment or secrets."
            )
            logger.error(error_msg)
            st.error(error_msg)
            raise ValueError(error_msg)

        # Initialize the embedding model and database service
        try:
            # Initialize embedding model first
            # Note: The API key should be already set globally by the settings_manager
            # But we provide it explicitly as a fallback
            embed_model = OpenAIEmbedding(
                api_key=SETTINGS.OPENAI_API_KEY,
                timeout=45.0,  # Increase timeout to 45 seconds
                max_retries=5,  # More retries for temporary failures
                retry_min_seconds=1,
                retry_max_seconds=60,
            )
            logger.info("OpenAIEmbedding initialized successfully.")

            # Then initialize the database service with the embedding model
            self.db_service = EnhancedDuckDBService(
                db_path=self.duckdb_settings["db_path"],
                embed_model=embed_model,
            )
            logger.info(
                f"EnhancedDuckDBService initialized with db_path: {self.duckdb_settings['db_path']}"
            )

        except Exception as e:
            error_msg = f"Failed to initialize services: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            raise ValueError(error_msg) from e

        # Initialize chat history in session state if needed
        if "messages" not in st.session_state:
            st.session_state.messages = []
            logger.debug("Initialized 'messages' in session state.")

    def _get_table_name(self, file_name: str | None = None, file_path: str | None = None) -> str:
        """Get a standardized table name from file name or path."""
        if file_name:
            return file_name.split(".")[0].lower().replace(" ", "_")
        if file_path:
            return os.path.basename(file_path).split(".")[0].lower().replace(" ", "_")
        return "user_data"

    def _load_from_file_path(self) -> bool:
        """Try to load data from file path in session state."""
        if "file_path" not in st.session_state or not st.session_state.file_path:
            return False

        file_path = st.session_state.file_path
        logger.debug(f"Found 'file_path' in session state: {file_path}")

        # Derive table name
        file_name = st.session_state.file_name if "file_name" in st.session_state else None
        table_name = self._get_table_name(file_name, file_path)
        logger.debug(f"Using table name: {table_name}")

        # Try direct file loading
        if self.db_service.load_file_directly(file_path, table_name):
            logger.info(f"Successfully loaded file into table: {table_name}")
            self.db_service.initialize()
            return True

        logger.warning("Failed to load file directly")
        return False

    def _load_from_dataframe(self) -> bool:
        """Try to load data from DataFrame in session state."""
        if "user_data" not in st.session_state or not isinstance(
            st.session_state.user_data, pd.DataFrame
        ):
            return False

        logger.debug("Using DataFrame loading method")
        main_df = st.session_state.user_data

        # Derive table name
        file_name = st.session_state.file_name if "file_name" in st.session_state else None
        table_name = self._get_table_name(file_name)

        # Fix DataFrame type issues
        for col in main_df.columns:
            if hasattr(main_df[col].dtype, "name") and "Int" in main_df[col].dtype.name:
                main_df[col] = main_df[col].astype("float64")
            if main_df[col].dtype == "object":
                try:
                    main_df[col] = main_df[col].astype("string")
                except (TypeError, ValueError) as e:
                    logger.error(f"Error converting column {col} to string: {e}")

        # Load the DataFrame
        if self.db_service.load_dataframe(main_df, table_name):
            logger.info(f"Successfully loaded DataFrame into table: {table_name}")
            self.db_service.initialize()
            return True

        return False

    def load_data_from_session(self) -> None:
        """
        Load data from Streamlit session state into the database.

        First tries to load from file path, then falls back to DataFrame loading.
        """
        logger.info("Attempting to load data from session state into DuckDB...")

        # Try loading from file path first, then from DataFrame
        if self._load_from_file_path() or self._load_from_dataframe():
            return

        # If we got here, no data was loaded
        logger.warning("No suitable data found in session state to load.")
        st.warning("No data was loaded. Please upload a file first.")

    def get_tables(self) -> list[str]:
        """
        Get list of available tables from the EnhancedDuckDBService.

        Returns
        -------
        list[str]
            List of table names managed by the service.
        """
        return self.db_service.tables

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query against the database.

        Parameters
        ----------
        query : str
            SQL query to execute

        Returns
        -------
        pd.DataFrame
            Query results as a pandas DataFrame

        Raises
        ------
        Exception
            If query execution fails
        """
        logger.info(f"Executing direct SQL query: {query[:100]}...")

        try:
            result_df = self.db_service.execute_query(query)
            logger.info(f"SQL query executed successfully, returned shape: {result_df.shape}")
            return result_df

        except Exception as e:
            logger.error(f"Error executing SQL query: {e}")
            st.error(f"Query execution error: {str(e)}")
            # Convert to a more user-friendly exception
            raise ValueError(f"SQL query execution failed: {str(e)}") from e

    def process_query(self, query: str) -> str:
        """
        Process a natural language query using EnhancedDuckDBService.

        Translates the user's natural language question into SQL, executes it,
        and returns a formatted response.

        Parameters
        ----------
        query : str
            Natural language query from the user

        Returns
        -------
        str
            Formatted response text with analysis results, or an error message.
        """
        logger.info(f"Processing natural language query: {query[:100]}...")
        tables = self.get_tables()
        if not tables:
            logger.warning("process_query called but no tables are loaded.")
            return "No data is loaded yet. Please upload your data first."

        # Determine complexity (simple placeholder for now)
        complexity: Literal["simple", "advanced"] = "advanced" if len(tables) > 1 else "simple"
        logger.debug(f"Using complexity: {complexity} for NL query.")

        try:
            # Use the service's process_query for natural language
            result = self.db_service.process_query(
                query=query, query_type="natural_language", complexity=complexity
            )
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            logger.error(f"Exception during process_query: {e}", exc_info=True)
            error_detail = f"\n```\n{traceback.format_exc()}\n```" if SETTINGS.DEBUG_MODE else ""
            return f"❌ An unexpected error occurred: {str(e)}{error_detail}"

        # Check if query was successful
        if not result.get("success"):
            error_message = result.get(
                "error", "An unknown error occurred during query processing."
            )
            logger.error(f"Failed to process natural language query: {error_message}")
            return f"❌ Error processing query: {error_message}"

        # Extract response data
        response_text = result.get("data", "No response text found.")
        sql_query = result.get("sql_query", "SQL query not available.")
        explanation = result.get("explanation", "")

        logger.info(f"Natural language query processed successfully. SQL: {sql_query}")

        # Format the response for the chat
        formatted_response = (
            f"**Analysis Result:**\n{response_text}\n\n---\n"
            f"*Executed SQL Query:*\n```sql\n{sql_query}\n```"
        )
        if explanation:
            formatted_response += f"\n*Explanation:* {explanation}"

        return formatted_response

    def get_chat_history(self) -> list[dict[str, str]]:
        """
        Get the chat history from session state.

        Returns
        -------
        list[dict[str, str]]
            List of message dictionaries with 'role' and 'content'
        """
        messages: list[dict[str, str]] = st.session_state.messages
        return messages

    def clear_chat_history(self) -> None:
        """Clear the chat history in session state."""
        st.session_state.messages = []
