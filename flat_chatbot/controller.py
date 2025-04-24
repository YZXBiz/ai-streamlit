"""
Controller module for the Data File/Table Chatbot application.

This module contains the AppController class which serves as the primary interface
between the UI components and the backend DuckDB service. It handles data loading,
query processing, and maintains application state.
"""

import os
import tempfile
import concurrent.futures
from typing import Any, Dict, List, Optional

import streamlit as st
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from flat_chatbot.config import SETTINGS
from flat_chatbot.services.duckdb_enhanced import EnhancedDuckDBService
from flat_chatbot.logger import get_logger

import os 
os.environ["OPENAI_API_KEY"] = SETTINGS.OPENAI_API_KEY

logger = get_logger(__name__)

class AppController:
    """
    Main application controller that coordinates between UI and data services.
    
    This class initializes the DuckDB service with OpenAI embeddings and LLM,
    manages file uploads and processing, handles query execution, and provides
    access to application state like tables and chat history.
    """
    
    def __init__(self) -> None:
        """Initialize the controller with the DuckDB service."""
        self.svc = self._init_service()
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        if "last_result" not in st.session_state:
            st.session_state.last_result = None
        logger.debug(f"AppController initialized, tables: {self.svc.tables if self.svc else 'None'}")

    def _init_service(self) -> Optional[EnhancedDuckDBService]:
        """Initialize the DuckDB service with OpenAI embeddings and LLM."""
        if not SETTINGS.OPENAI_API_KEY:
            st.error("OPENAI_API_KEY not set")
            return None
        embed = OpenAIEmbedding(
                                model=SETTINGS.EMBEDDING_MODEL,)
        llm = OpenAI(
                     model=SETTINGS.OPENAI_MODEL, 
                     temperature=SETTINGS.OPENAI_TEMPERATURE, 
                     )
        return EnhancedDuckDBService(
            embed_model=embed,
            llm_model=llm,
            memory_type=SETTINGS.MEMORY_TYPE,
            token_limit=SETTINGS.TOKEN_LIMIT,
            
            # prompt templates
            user_text_to_sql_prompt=SETTINGS.USER_TEXT_TO_SQL_PROMPT,
            user_response_synthesis_prompt=SETTINGS.USER_RESPONSE_SYNTHESIS_PROMPT,
            user_refine_synthesis_prompt=SETTINGS.USER_REFINE_SYNTHESIS_PROMPT,
        )

    def upload_files(self, files: List[Any]) -> bool:
        """
        Process uploaded files and load them into DuckDB tables.
        
        Parameters
        ----------
        files : list
            List of Streamlit UploadedFile objects
            
        Returns
        -------
        bool
            True if any new files were processed, False otherwise
        """
        from flat_chatbot.ui.upload import table_exists

        logger.debug(f"Beginning upload_files with {len(files) if files else 0} files")
        logger.debug(f"Tables before upload: {self.svc.tables}")
        
        any_new = False
        for uf in files or []:
            fid = f"{uf.name}_{uf.size}"
            if fid in st.session_state.processed_files:
                continue
            tbl = uf.name.rsplit(".", 1)[0].replace(" ", "_")
            if table_exists(self.svc, tbl):
                st.sidebar.info(f"Skipping existing {tbl}")
                st.session_state.processed_files.add(fid)
                continue
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix="." + uf.name.rsplit(".", 1)[1])
            tmp.write(uf.getbuffer())
            tmp.close()
            ok = self.svc.load_file_directly(tmp.name, tbl)
            os.unlink(tmp.name)
            if ok:
                st.sidebar.success(f"Loaded {tbl}")
                any_new = True
                st.session_state.processed_files.add(fid)
            else:
                st.sidebar.error(f"Failed {tbl}")
        
        if any_new:
            logger.debug("New files loaded, calling initialize()...")
            self.svc.initialize()
            
        logger.debug(f"Tables after upload: {self.svc.tables}")
        return any_new

    def ask(self, query: str, complexity: str) -> Dict[str, Any]:
        """
        Process a natural language query with timeout handling.
        
        Parameters
        ----------
        query : str
            The natural language query to process
        complexity : str
            Complexity level for query processing
            
        Returns
        -------
        dict
            Results dictionary with success flag and data or error message
        """
        def _call() -> Dict[str, Any]:
            # This is potentially problematic due to thread-safety issues with DuckDB/state,
            # but restoring as requested.
            return self.svc.process_query(query, "natural_language", complexity)

        with concurrent.futures.ThreadPoolExecutor() as ex:
            future = ex.submit(_call)
            try:
                # Add debug print before returning result
                result = future.result(timeout=SETTINGS.QUERY_TIMEOUT)
                tables_after_query = self.svc.get_tables() if self.svc else 'SVC NONE'
                logger.debug(f'Tables after query in controller.ask: {tables_after_query}')
                return result
            except concurrent.futures.TimeoutError:
                return {"success": False, "error": f"Timeout after {SETTINGS.QUERY_TIMEOUT}s"}
            except Exception as e:
                # Catch potential errors during execution in thread
                logger.exception("Error processing query in controller.ask thread")
                return {"success": False, "error": f"An error occurred: {str(e)}"}

    def clear_all(self) -> None:
        """Remove all loaded tables and reset the processed files state."""
        self.svc.clear_data()
        st.session_state.processed_files.clear()

    def get_tables(self) -> List[str]:
        """Get list of all available tables."""
        return self.svc.tables

    def get_schema(self) -> Dict[str, Any]:
        """Get schema information for all tables."""
        return self.svc.get_schema_info()

    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent query result."""
        return st.session_state.last_result

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.svc.clear_chat_history()
        
    def get_table_list(self) -> List[str]:
        """
        Get list of all available tables.
        
        Returns
        -------
        list
            List of table names as strings
        """
        if self.svc is None:
            return []
        tables = self.svc.get_tables()
        logger.debug(f"get_table_list returning: {tables}")
        return tables
    
    def get_table_schema(self, table: str) -> Dict[str, str]:
        """
        Get schema information for a specific table.
        
        Parameters
        ----------
        table : str
            Name of the table to get schema for
            
        Returns
        -------
        dict
            Dictionary mapping column names to their data types
        """
        if self.svc is None:
            return {}
            
        try:
            # Get column information
            columns_df = self.svc.execute_query(f"DESCRIBE {table}")
            
            # Create a dictionary mapping column names to types
            schema = {}
            for _, row in columns_df.iterrows():
                schema[row['column_name']] = row['column_type']
                
            return schema
        except Exception as e:
            st.error(f"Error getting table schema for {table}: {str(e)}")
            return {}
    
    def execute_query(self, query: str) -> Any:
        """
        Execute a SQL query against the database.
        
        Parameters
        ----------
        query : str
            SQL query to execute
            
        Returns
        -------
        pandas.DataFrame
            DataFrame with query results
        """
        if self.svc is None:
            return None
            
        try:
            return self.svc.execute_query(query)
        except Exception as e:
            st.error(f"Error executing query: {str(e)}")
            return None
    
    def import_dataframe(self, df: Any, table_name: str) -> bool:
        """
        Import a pandas DataFrame into the database.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to import
        table_name : str
            Name of the table to create
            
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        if self.svc is None:
            return False
            
        try:
            success = self.svc.load_dataframe(df, table_name)
            if success:
                logger.debug(f"Successfully imported DataFrame to {table_name}")
                # Signal the UI to rerun and pick up new tables
                st.session_state.just_uploaded = True
            return success
        except Exception as e:
            logger.exception(f"Error importing DataFrame to {table_name}")
            st.error(f"Error importing DataFrame: {str(e)}")
            return False
