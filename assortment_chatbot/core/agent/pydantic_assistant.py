"""PydanticAI-powered data chat assistant.

This module integrates the DuckDB service, data chat agent, and result interpreter
to provide a complete data chat experience for users.
"""

import hashlib
import os
import time
from typing import Any

import pandas as pd
import streamlit as st

from assortment_chatbot.config.settings import get_settings
from assortment_chatbot.core.agent.data_chat_agent import (
    execute_query_and_analyze,
    process_query,
    validate_and_refine_sql,
)
from assortment_chatbot.core.agent.result_interpreter import (
    interpret_results,
    verify_interpretation,
)
from assortment_chatbot.services.duckdb_service import DuckDBService
from assortment_chatbot.utils.logging import get_logger

# Setup logging
logger = get_logger(__name__)

# Get settings
settings = get_settings()

# Timeout for query processing (in seconds) from settings
QUERY_TIMEOUT = settings.agent_settings["query_timeout"]


class PydanticAssistant:
    """PydanticAI-powered data assistant for SQL generation and analysis.

    This class integrates all the components needed for a data chat experience:
    - DuckDB for data storage and querying
    - PydanticAI agents for query processing and result interpretation
    - Enhanced pipeline with context collection, validation, and memory
    """

    def __init__(self) -> None:
        """Initialize the data chat assistant."""
        # Initialize with settings
        self.db_service = DuckDBService()
        self.current_table: str | None = None
        self.debug_mode = settings.DEBUG_MODE

        # Ensure required session state variables exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if "last_query_result" not in st.session_state:
            st.session_state.last_query_result = None

        # Add conversation memory
        if "conversation_memory" not in st.session_state:
            st.session_state.conversation_memory = []

    def load_data_from_session(self) -> bool:
        """Load data from the session state if available.

        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        if "user_data" in st.session_state and isinstance(st.session_state.user_data, pd.DataFrame):
            try:
                # Clear any existing data
                self.db_service.clear_data()

                # Load the DataFrame into DuckDB
                df = st.session_state.user_data
                # Use a default table name or extract from filename if available
                table_name = "user_data"
                if "file_name" in st.session_state:
                    # Extract base name without extension
                    base_name = os.path.splitext(st.session_state.file_name)[0]
                    # Clean up the name to be SQL-friendly
                    table_name = "".join(c if c.isalnum() else "_" for c in base_name)

                # Store the data
                success = self.db_service.load_dataframe(df, table_name)
                if success:
                    self.current_table = table_name

                    # Debug output if enabled
                    if self.debug_mode:
                        st.sidebar.write(f"Debug: Loaded table '{table_name}' with {len(df)} rows")

                return success
            except Exception:
                logger.error("Error loading data from session", exc_info=True)
                return False
        return False

    def get_tables(self) -> list[str]:
        """Get list of available tables.

        Returns:
            List[str]: List of table names
        """
        schema_info = self.db_service.get_schema_info()
        return schema_info.get("tables", [])

    def _collect_context(self, query: str) -> dict:
        """Collect context information for query processing.

        Args:
            query: The user's query

        Returns:
            Dict with context information
        """
        context = {
            "schema_info": self.db_service.get_schema_info(),
            "data_statistics": self.db_service.get_data_statistics(self.current_table),
            "conversation_history": st.session_state.conversation_memory[-5:]
            if len(st.session_state.conversation_memory) > 0
            else [],
            "current_query": query,
        }

        # Add data sample if available
        try:
            context["data_sample"] = self.db_service.execute_query(
                f"SELECT * FROM {self.current_table} LIMIT 5"
            )
        except Exception:
            context["data_sample"] = None

        return context

    def process_user_query(self, query: str) -> dict[str, Any]:
        """Process a natural language query from the user with enhanced pipeline.

        Enhanced pipeline includes:
        1. Context collection
        2. SQL generation
        3. SQL validation and refinement
        4. Query execution
        5. Result interpretation and verification
        6. Response generation with conversation memory

        Args:
            query: Natural language query from the user

        Returns:
            Dict[str, Any]: Dictionary with results and interpretation
        """
        # Ensure data is loaded
        if not self.current_table:
            loaded = self.load_data_from_session()
            if not loaded:
                return {
                    "success": False,
                    "error": "No data has been loaded. Please upload data first.",
                }

        try:
            # Debug info
            if self.debug_mode:
                st.sidebar.write(f"Debug: Processing query: {query}")
                st.sidebar.write(f"Debug: Using SQL model: {agent_settings['sql_agent_model']}")

            # Generate a unique key for this query
            query_hash = hashlib.md5(query.encode()).hexdigest()
            query_key = f"query_start_time_{query_hash}"

            # Set start time for timeout tracking if not already set
            if query_key not in st.session_state:
                st.session_state[query_key] = time.time()

            start_time = st.session_state[query_key]

            # Check if we've already exceeded the timeout
            if time.time() - start_time > QUERY_TIMEOUT:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]

                return {
                    "success": False,
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts.",
                }

            # 1. Collect context for the query
            context = self._collect_context(query)

            # 2. Process query to generate SQL with context
            query_result = process_query(query, self.db_service, self.current_table, context)

            # Check timeout after SQL generation
            if time.time() - start_time > QUERY_TIMEOUT:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]

                return {
                    "success": False,
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts.",
                }

            # Debug SQL
            if self.debug_mode:
                st.sidebar.code(query_result["sql"], language="sql")

            # 3. Validate and refine SQL
            refined_result = validate_and_refine_sql(
                query_result["sql"], self.db_service, self.current_table
            )

            # Use refined SQL if available
            final_sql = refined_result.get("refined_sql", query_result["sql"])

            if self.debug_mode and refined_result.get("refined_sql"):
                st.sidebar.write("SQL was refined:")
                st.sidebar.code(final_sql, language="sql")

            # 4. Execute the generated SQL
            execution_result = execute_query_and_analyze(final_sql, self.db_service)

            # Check timeout after SQL execution
            if time.time() - start_time > QUERY_TIMEOUT:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]

                return {
                    "success": False,
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts.",
                }

            # Check if execution was successful
            if not execution_result["success"]:
                # Try fallback with simpler query if appropriate
                if (
                    "syntax error" in execution_result["error"].lower()
                    or "not found" in execution_result["error"].lower()
                ):
                    simplified_sql = f"SELECT * FROM {self.current_table} LIMIT 10"
                    fallback_result = execute_query_and_analyze(simplified_sql, self.db_service)

                    if fallback_result["success"]:
                        # We got a result with fallback query
                        if self.debug_mode:
                            st.sidebar.write("Using fallback query:")
                            st.sidebar.code(simplified_sql, language="sql")

                        # 5. Interpret the fallback results
                        interpretation = interpret_results(
                            original_query=query,
                            sql_query=simplified_sql,
                            result_df=fallback_result["result_data"],
                            is_fallback=True,
                        )

                        # Update conversation memory
                        self._update_conversation_memory(query, interpretation, simplified_sql)

                        # Clear the session state for this query
                        if query_key in st.session_state:
                            del st.session_state[query_key]

                        return {
                            "success": True,
                            "sql": simplified_sql,
                            "explanation": f"I couldn't execute your specific query due to an error ({execution_result['error']}), but here's some general information from the data:",
                            "interpretation": interpretation,
                            "result_data": fallback_result["result_data"],
                            "stats": fallback_result.get("stats", {}),
                            "is_fallback": True,
                        }

                # If fallback also failed or wasn't attempted, return the error
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]

                return {
                    "success": False,
                    "error": execution_result["error"],
                    "sql": final_sql,
                    "explanation": query_result["explanation"],
                }

            # 5. Interpret the results
            interpretation = interpret_results(
                original_query=query,
                sql_query=final_sql,
                result_df=execution_result["result_data"],
            )

            # 6. Verify the interpretation for accuracy
            verification_result = verify_interpretation(
                interpretation=interpretation,
                result_df=execution_result["result_data"],
                sql_query=final_sql,
            )

            if not verification_result["verified"]:
                # Make adjustments to interpretation if needed
                interpretation = verification_result["corrected_interpretation"]

                if self.debug_mode:
                    st.sidebar.write("Interpretation was corrected during verification")

            # Final timeout check
            if time.time() - start_time > QUERY_TIMEOUT:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]

                return {
                    "success": False,
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts.",
                }

            # 7. Store in conversation memory for context in future queries
            self._update_conversation_memory(query, interpretation, final_sql)

            # Clear query timeout tracking
            if query_key in st.session_state:
                del st.session_state[query_key]

            # Store the result in session state for potential download
            st.session_state.last_query_result = execution_result["result_data"]

            # Return the complete result
            return {
                "success": True,
                "sql": final_sql,
                "explanation": query_result["explanation"],
                "interpretation": interpretation,
                "result_data": execution_result["result_data"],
                "stats": execution_result.get("stats", {}),
            }

        except Exception as e:
            logger.error("Error processing query", exc_info=True)
            # Clear query timeout tracking
            if query_key in st.session_state:
                del st.session_state[query_key]

            return {
                "success": False,
                "error": f"Error processing query: {str(e)}",
            }

    def _update_conversation_memory(self, query: str, response: str, sql: str) -> None:
        """Update the conversation memory with the latest exchange.

        Args:
            query: The user's query
            response: The system's response
            sql: The SQL that was executed
        """
        memory_entry = {"query": query, "response": response, "sql": sql, "timestamp": time.time()}

        # Add to conversation memory, keeping the last 10 exchanges
        st.session_state.conversation_memory.append(memory_entry)
        if len(st.session_state.conversation_memory) > 10:
            st.session_state.conversation_memory.pop(0)

    def download_last_result(self, file_format: str = "csv") -> bytes | None:
        """Prepare the last query result for download in the specified format.

        Args:
            file_format: The format to download the data in (csv, xlsx, json)

        Returns:
            bytes: The file content as bytes, or None if no result is available
        """
        if (
            "last_query_result" not in st.session_state
            or st.session_state.last_query_result is None
        ):
            return None

        df = st.session_state.last_query_result

        try:
            if file_format == "csv":
                return df.to_csv(index=False).encode("utf-8")
            elif file_format == "xlsx":
                import io

                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                return buffer.getvalue()
            elif file_format == "json":
                return df.to_json(orient="records").encode("utf-8")
            else:
                logger.error(f"Unsupported file format: {file_format}")
                return None
        except Exception:
            logger.error(f"Error converting dataframe to {file_format}", exc_info=True)
            return None

    def _add_to_chat_history(
        self, role: str, content: str, result: dict[str, Any] | None = None
    ) -> None:
        """Add a message to the chat history.

        Args:
            role: The role of the sender (user or assistant)
            content: The message content
            result: Optional result data to store with the message
        """
        st.session_state.chat_history.append({"role": role, "content": content, "result": result})

    def get_chat_history(self) -> list[dict[str, Any]]:
        """Get the current chat history.

        Returns:
            List of chat messages with roles and content
        """
        return st.session_state.chat_history

    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        st.session_state.chat_history = []
        # Also clear conversation memory for a fresh start
        st.session_state.conversation_memory = []

    def process_query(self, query: str) -> str:
        """Process a user query and return a response.

        This is a convenience method that processes the query and formats
        the response for display.

        Args:
            query: The user's query

        Returns:
            Formatted response to display to the user
        """
        # Process the query
        result = self.process_user_query(query)

        # Store in chat history
        self._add_to_chat_history("user", query)

        if result["success"]:
            # Format successful response
            response = result["interpretation"]
            self._add_to_chat_history("assistant", response, result)
            return response
        else:
            # Format error response
            error_msg = f"Sorry, I couldn't process your query: {result['error']}"
            self._add_to_chat_history("assistant", error_msg)
            return error_msg
