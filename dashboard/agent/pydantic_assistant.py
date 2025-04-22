"""PydanticAI-powered data chat assistant.

This module integrates the DuckDB service, data chat agent, and result interpreter
to provide a complete data chat experience for users.
"""

import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List
import time
import hashlib

from dashboard.data.duckdb_service import DuckDBService
from dashboard.agent.core.data_chat_agent import process_query, execute_query_and_analyze
from dashboard.agent.core.result_interpreter import interpret_results
from dashboard.settings import get_settings

# Get settings
settings = get_settings()
agent_settings = settings.agent_settings

# Timeout for query processing (in seconds) from settings
QUERY_TIMEOUT = agent_settings["query_timeout"]


class PydanticAssistant:
    """PydanticAI-powered data assistant for SQL generation and analysis.
    
    This class integrates all the components needed for a data chat experience:
    - DuckDB for data storage and querying
    - PydanticAI agents for query processing and result interpretation
    """
    
    def __init__(self):
        """Initialize the data chat assistant."""
        # Initialize with settings
        self.db_service = DuckDBService()
        self.current_table: Optional[str] = None
        self.debug_mode = settings.debug_mode
        
        # Ensure required session state variables exist
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        if "last_query_result" not in st.session_state:
            st.session_state.last_query_result = None
    
    def load_data_from_session(self) -> bool:
        """Load data from the session state if available.
        
        Returns:
            True if data was loaded successfully, False otherwise
        """
        if "user_data" in st.session_state and isinstance(st.session_state.user_data, pd.DataFrame):
            # Clear any existing data
            self.db_service.clear_data()
            
            # Load the DataFrame into DuckDB
            df = st.session_state.user_data
            # Use a default table name or extract from filename if available
            table_name = "user_data"
            if "file_name" in st.session_state:
                # Extract base name without extension
                import os
                base_name = os.path.splitext(st.session_state.file_name)[0]
                # Clean up the name to be SQL-friendly
                table_name = ''.join(c if c.isalnum() else '_' for c in base_name)
            
            # Store the data
            success = self.db_service.load_dataframe(df, table_name)
            if success:
                self.current_table = table_name
                
                # Debug output if enabled
                if self.debug_mode:
                    st.sidebar.write(f"Debug: Loaded table '{table_name}' with {len(df)} rows")
                
            return success
        return False
    
    def get_tables(self) -> List[str]:
        """Get list of available tables.
        
        Returns:
            List of table names
        """
        schema_info = self.db_service.get_schema_info()
        return schema_info.get("tables", [])
    
    def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query from the user.
        
        Args:
            query: Natural language query from the user
            
        Returns:
            Dictionary with results and interpretation
        """
        # Ensure data is loaded
        if not self.current_table:
            loaded = self.load_data_from_session()
            if not loaded:
                return {
                    "success": False,
                    "error": "No data has been loaded. Please upload data first."
                }
        
        try:
            # Debug info
            if self.debug_mode:
                st.sidebar.write(f"Debug: Processing query: {query}")
                st.sidebar.write(f"Debug: Using SQL model: {settings.agent.sql_agent_model}")
            
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
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts."
                }
            
            # 1. Process query to generate SQL
            query_result = process_query(query, self.db_service, self.current_table)
            
            # Check timeout after SQL generation
            if time.time() - start_time > QUERY_TIMEOUT:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]
                
                return {
                    "success": False,
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts."
                }
            
            # Debug SQL
            if self.debug_mode:
                st.sidebar.code(query_result["sql"], language="sql")
            
            # 2. Execute the generated SQL
            execution_result = execute_query_and_analyze(query_result["sql"], self.db_service)
            
            # Check timeout after SQL execution
            if time.time() - start_time > QUERY_TIMEOUT:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]
                
                return {
                    "success": False,
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts."
                }
            
            # Check if execution was successful
            if not execution_result["success"]:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]
                
                return {
                    "success": False,
                    "error": execution_result["error"],
                    "sql": query_result["sql"],
                    "explanation": query_result["explanation"]
                }
            
            # 3. Interpret the results
            interpretation = interpret_results(
                original_query=query,
                sql_query=query_result["sql"],
                result_df=execution_result["result_data"]
            )
            
            # Final timeout check
            if time.time() - start_time > QUERY_TIMEOUT:
                # Clear the session state for this query
                if query_key in st.session_state:
                    del st.session_state[query_key]
                
                return {
                    "success": False,
                    "error": f"Query processing timed out after {QUERY_TIMEOUT} seconds. Please try a simpler query or break it into smaller parts."
                }
            
            # 4. Store result in session state for potential download
            full_result = {
                "success": True,
                "original_query": query,
                "sql": query_result["sql"],
                "explanation": query_result["explanation"],
                "visualization_type": query_result["visualization_type"],
                "transform_explanation": query_result["transform_explanation"],
                "result_data": execution_result["result_data"],
                "analysis": execution_result["analysis"],
                "interpretation": interpretation
            }
            
            # Update last query result for download
            st.session_state.last_query_result = full_result
            
            # 5. Add to chat history
            self._add_to_chat_history(
                "user", 
                query,
                full_result
            )
            
            # Clear the session state for this query as it completed successfully
            if query_key in st.session_state:
                del st.session_state[query_key]
            
            return full_result
            
        except Exception as e:
            # Generate the query key for cleanup
            query_hash = hashlib.md5(query.encode()).hexdigest()
            query_key = f"query_start_time_{query_hash}"
            
            # Clear the session state for this query
            if query_key in st.session_state:
                del st.session_state[query_key]
            
            error_message = f"Error processing query: {str(e)}"
            # Show more details in debug mode
            if self.debug_mode:
                import traceback
                st.sidebar.error(f"Debug: Exception details: {traceback.format_exc()}")
                
            return {
                "success": False,
                "error": error_message
            }
    
    def download_last_result(self, file_format: str = "csv") -> Optional[bytes]:
        """Generate a downloadable file of the last query result.
        
        Args:
            file_format: Format of the output file (csv, xlsx, json)
            
        Returns:
            Bytes of the file or None if no result is available
        """
        if "last_query_result" not in st.session_state or not st.session_state.last_query_result:
            return None
        
        result = st.session_state.last_query_result
        if not result["success"] or "result_data" not in result:
            return None
        
        df = result["result_data"]
        
        # Generate file in the requested format
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
            return None
    
    def _add_to_chat_history(self, role: str, content: str, result: Optional[Dict[str, Any]] = None) -> None:
        """Add a message to the chat history.
        
        Args:
            role: 'user' or 'assistant'
            content: Text content of the message
            result: Optional structured result data for assistant messages
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if result:
            message["result"] = result
        
        st.session_state.chat_history.append(message)
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get the chat history.
        
        Returns:
            List of chat messages
        """
        return st.session_state.chat_history
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        st.session_state.chat_history = []
        st.session_state.last_query_result = None

    def process_query(self, query: str) -> str:
        """Process a user query and return a response.
        
        This is a simplified wrapper around process_user_query that returns
        a markdown-formatted string suitable for display in the chat interface.
        
        Args:
            query: The user's natural language query
            
        Returns:
            A markdown-formatted response string
        """
        result = self.process_user_query(query)
        
        if not result["success"]:
            return f"❌ Error: {result['error']}"
            
        # Format the successful response
        interpretation = result["interpretation"]
        
        # Build a markdown response
        response = f"""
{interpretation['summary']}

**Key Insights:**
"""
        
        # Add insights if available
        for insight in interpretation.get("insights", []):
            response += f"- {insight}\n"
            
        # Add recommendations if available
        if interpretation.get("recommendations", []):
            response += "\n**Recommendations:**\n"
            for rec in interpretation["recommendations"]:
                response += f"- {rec}\n"
                
        return response


def display_chat_interface():
    """Display a Streamlit chat interface with the data chat assistant."""
    # Initialize the assistant
    if "data_assistant" not in st.session_state:
        st.session_state.data_assistant = PydanticAssistant()
    
    # Get assistant instance
    assistant = st.session_state.data_assistant
    
    # Make sure data is loaded
    assistant.load_data_from_session()
    
    # Display tables loaded
    available_tables = assistant.get_tables()
    if available_tables:
        st.sidebar.success(f"Data loaded: {', '.join(available_tables)}")
    else:
        st.sidebar.warning("No data loaded. Please upload data first.")
    
    # Display settings info if in debug mode
    if settings.debug_mode:
        with st.sidebar.expander("Debug: Settings Info", expanded=False):
            st.write(f"Environment: {settings.environment}")
            st.write(f"SQL Agent Model: {settings.agent.sql_agent_model}")
            st.write(f"Interpreter Model: {settings.agent.interpreter_model}")
            st.write(f"Temperature: {settings.agent.temperature}")
            st.write(f"DuckDB Path: {settings.duckdb.db_path}")
    
    # Chat history
    chat_history = assistant.get_chat_history()
    
    # Clear chat button
    if st.sidebar.button("Clear Chat History"):
        assistant.clear_chat_history()
        st.rerun()
    
    # Download last result
    if "last_query_result" in st.session_state and st.session_state.last_query_result:
        st.sidebar.subheader("Download Results")
        file_format = st.sidebar.selectbox("Format", ["csv", "xlsx", "json"], index=0)
        if st.sidebar.button(f"Download as {file_format.upper()}"):
            data = assistant.download_last_result(file_format)
            if data:
                st.sidebar.download_button(
                    label=f"Download {file_format.upper()}",
                    data=data,
                    file_name=f"query_result.{file_format}",
                    mime=_get_mime_type(file_format)
                )
    
    # Display chat messages
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # If it's an assistant message with results, show them
            if message["role"] == "assistant" and "result" in message and message["result"]["success"]:
                result = message["result"]
                
                # Display a preview of the data
                with st.expander("View Data", expanded=False):
                    st.dataframe(result["result_data"], use_container_width=True)
                
                # Display SQL query
                with st.expander("SQL Query", expanded=False):
                    st.code(result["sql"], language="sql")
                
                # Display insights
                if "interpretation" in result:
                    interpretation = result["interpretation"]
                    with st.expander("Insights", expanded=True):
                        st.write(interpretation["summary"])
                        if interpretation["insights"]:
                            st.subheader("Key Insights")
                            for insight in interpretation["insights"]:
                                st.markdown(f"- {insight}")
                        if interpretation["recommendations"]:
                            st.subheader("Recommendations")
                            for rec in interpretation["recommendations"]:
                                st.markdown(f"- {rec}")
    
    # Input for new message
    if query := st.chat_input("Ask about your data..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Process the query and display the result
        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                result = assistant.process_user_query(query)
                
                if result["success"]:
                    # Display the interpretation summary
                    interpretation = result["interpretation"]
                    st.markdown(interpretation["summary"])
                    
                    # Show dataframe preview
                    st.dataframe(result["result_data"], use_container_width=True)
                    
                    # Add to chat history
                    assistant._add_to_chat_history(
                        "assistant",
                        interpretation["summary"],
                        result
                    )
                else:
                    # Display error message
                    st.error(result["error"])
                    
                    # Add to chat history
                    assistant._add_to_chat_history(
                        "assistant",
                        f"Error: {result['error']}"
                    )


def _get_mime_type(file_format: str) -> str:
    """Get the MIME type for a file format.
    
    Args:
        file_format: File format (csv, xlsx, json)
        
    Returns:
        MIME type string
    """
    mime_types = {
        "csv": "text/csv",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "json": "application/json"
    }
    return mime_types.get(file_format, "text/plain") 