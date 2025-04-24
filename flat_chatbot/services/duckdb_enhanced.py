"""
Enhanced DuckDB service module.

This module extends the base DuckDB service with natural language query support
using LlamaIndex and OpenAI embeddings.
"""

from typing import Any, Literal

import pandas as pd
import streamlit as st
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.struct_store import (
    NLSQLTableQueryEngine,
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.indices.struct_store.sql_query import SQLDatabase
from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.schema import Node
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from sqlalchemy import create_engine

from flat_chatbot.logger import get_logger
from flat_chatbot.services.duckdb_base import DuckDBService
from llama_index.core.prompts.default_prompts import (
    DEFAULT_TEXT_TO_SQL_PROMPT,
    DEFAULT_REFINE_PROMPT
)
from llama_index.core.indices.struct_store.sql_query import (
    DEFAULT_RESPONSE_SYNTHESIS_PROMPT_V2,
)
from llama_index.core.prompts import BasePromptTemplate

logger = get_logger(__name__)


class EnhancedDuckDBService(DuckDBService):
    """Enhanced DuckDB service with LlamaIndex integration for NL2SQL.

    This class extends the base DuckDBService with LlamaIndex capabilities
    to support natural language to SQL conversion, query retrieval based on
    schema understanding, and contextual responses.
    """

    def __init__(
        self,
        embed_model: OpenAIEmbedding,
        llm_model: OpenAI,
        db_path: str | None = None,
        memory_type: Literal["simple", "summary"] = "simple",
        token_limit: int = 3000,
        chat_store: BaseChatStore | None = None,
        chat_store_key: str = "default_session",
        user_text_to_sql_prompt: BasePromptTemplate | None = None,
        user_response_synthesis_prompt: BasePromptTemplate | None = None,
        user_refine_synthesis_prompt: BasePromptTemplate | None = None,
    ):
        """Initialize the enhanced DuckDB service.

        Args:
            embed_model: OpenAI embedding model for LlamaIndex
            llm_model: LlamaIndex OpenAI model interface (not the openai client)
            db_path: Optional path to DuckDB database file. If None, an in-memory
                    database will be used.
            memory_type: Type of chat memory to use ("simple" or "summary")
            token_limit: Maximum number of tokens to store in memory
            chat_store: Optional custom chat store for persistence
            chat_store_key: Key to identify this chat session
            user_text_to_sql_prompt: Custom prompt template for text-to-SQL conversion
            user_response_synthesis_prompt: Custom prompt template for response synthesis
            user_refine_synthesis_prompt: Custom prompt template for refinement
        """
        # Initialize the base DuckDBService
        super().__init__(db_path)
        
        # Store the database path for reuse with SQLAlchemy
        self._db_path = db_path or ":memory:"

        # Initialize LlamaIndex components
        self.table_schemas = {}
        self.simple_query_engine = None
        self.advanced_query_engine = None
        self.table_index = None
        self.embed_model = embed_model
        self.llm_model = llm_model
        
        # prompt templates - ensure they're proper BasePromptTemplate instances
        self.text_to_sql_prompt = user_text_to_sql_prompt or DEFAULT_TEXT_TO_SQL_PROMPT
        self.response_synthesis_prompt = user_response_synthesis_prompt or DEFAULT_RESPONSE_SYNTHESIS_PROMPT_V2
        self.refine_synthesis_prompt = user_refine_synthesis_prompt or DEFAULT_REFINE_PROMPT

        # Initialize chat memory
        self._init_chat_memory(memory_type, token_limit, chat_store, chat_store_key)

    def _init_chat_memory(
        self,
        memory_type: str,
        token_limit: int,
        chat_store: BaseChatStore | None,
        chat_store_key: str,
    ) -> None:
        """Initialize the chat memory buffer.

        Args:
            memory_type: Type of memory to use
            token_limit: Maximum tokens to store
            chat_store: Optional custom chat store
            chat_store_key: Session identifier
        """
        # Use provided chat store or create a default one
        if chat_store is None:
            chat_store = SimpleChatStore()

        # Create appropriate memory buffer based on type
        if memory_type == "summary":
            self.memory = ChatSummaryMemoryBuffer.from_defaults(
                token_limit=token_limit, chat_store=chat_store, chat_store_key=chat_store_key
            )
        else:  # default to simple
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=token_limit, chat_store=chat_store, chat_store_key=chat_store_key
            )

        logger.info(f"Initialized {memory_type} chat memory with token limit {token_limit}")

    def initialize(self) -> None:
        """Initialize LlamaIndex components based on current database state."""
        logger.debug(f"INITIALIZE start: self.tables = {self.tables}")
        # Get table schemas directly from DuckDB
        self._get_table_schemas()

        # Initialize query engines if we have tables
        if self.tables:
            logger.debug(f"INITIALIZE before _init_query_engines: self.tables = {self.tables}")
            self._init_query_engines()
            logger.debug(f"INITIALIZE after _init_query_engines: self.tables = {self.tables}")
            logger.info("Initialized LlamaIndex query engines")
        else:
            logger.info("No tables available for LlamaIndex initialization")
        logger.debug(f"INITIALIZE end: self.tables = {self.tables}")

    def _get_table_schemas(self) -> None:
        """Get schema information for all tables in DuckDB."""
        self.table_schemas = {}

        for table in self.tables:
            # Get column information
            columns_df = self.execute_query(f"DESCRIBE {table}")
            # Store schema information for LlamaIndex
            column_names = columns_df["column_name"].tolist()
            column_types = columns_df["column_type"].tolist()
            self.table_schemas[table] = {
                "name": table,
                "columns": column_names,
                "types": column_types,
            }

    def _init_query_engines(self) -> None:
        """Initialize LlamaIndex query engines for natural language queries."""
        if not self.table_schemas:
            logger.warning("No table schemas available for query engine initialization")
            return

        # Create schema nodes for vector search
        schema_nodes = []
        for table_name, schema_info in self.table_schemas.items():
            # Create a detailed description of the table for embedding
            schema_text = (
                f"Table '{table_name}' with description: collection of {table_name} data. "
                f"Columns: {', '.join([f'{col} ({col_type})' for col, col_type in zip(schema_info['columns'], schema_info['types'], strict=True)])}"
            )
            # Create Node with text directly instead of using MediaResource
            node = Node(text=schema_text, metadata={"table_name": table_name})
            schema_nodes.append(node)

        # Create vector index from schema nodes with explicit embedding model
        self.table_index = VectorStoreIndex(schema_nodes, embed_model=self.embed_model)

        # CRITICAL FIX: Use the same database path for SQLAlchemy as for DuckDB connection
        try:
            # Create SQLAlchemy engine using the SAME database path as the DuckDB connection
            db_url = f"duckdb:///{self._db_path}"
            logger.debug(f"Creating SQLAlchemy engine with URL: {db_url}")
            engine = create_engine(db_url)
            
            # For in-memory databases we still need to explicitly copy tables
            # This ensures tables are accessible to both connections
            if self._db_path == ":memory:":
                for table_name in self.tables:
                    df = self.execute_query(f"SELECT * FROM {table_name}")
                    df.to_sql(table_name, engine, index=False, if_exists="replace")
                    logger.debug(f"Copied in-memory table {table_name} to SQLAlchemy engine")
            
            # Initialize SQLDatabase with the SQLAlchemy engine
            sql_database = SQLDatabase(engine=engine, include_tables=list(self.table_schemas.keys()))
            logger.debug(f"Created SQLDatabase with tables: {', '.join(self.table_schemas.keys())}")
        except Exception as e:
            # If anything fails, try the absolute simplest approach
            logger.warning(f"Standard approach failed: {e}, trying fallback")
            
            # Direct fallback approach - create a completely separate simple engine
            fallback_db_path = self._db_path if self._db_path != ":memory:" else ""
            backup_engine = create_engine(f"duckdb:///{fallback_db_path}")
            logger.debug(f"Created fallback SQLAlchemy engine with path: {fallback_db_path}")
            
            # Copy all tables with full data (only needed for in-memory DBs)
            if self._db_path == ":memory:":
                for table_name in self.tables:
                    logger.debug(f"Fallback: Copying {table_name} to backup engine")
                    df = self.execute_query(f"SELECT * FROM {table_name}")
                    df.to_sql(table_name, backup_engine, index=False, if_exists="replace")
            
            # Initialize with fallback engine
            sql_database = SQLDatabase(engine=backup_engine, include_tables=list(self.table_schemas.keys()))

        # Initialize simple query engine for natural language queries
        # CRITICAL FIX: Pass the LlamaIndex LLM model, not the OpenAI client
        self.simple_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=list(self.table_schemas.keys()),
            llm=self.llm_model,  # Now using the correct LlamaIndex LLM model
            embed_model=self.embed_model,
            text_to_sql_prompt=self.text_to_sql_prompt,
            response_synthesis_prompt=self.response_synthesis_prompt,
            refine_synthesis_prompt=self.refine_synthesis_prompt,
        )
        logger.debug(f"Initialized simple query engine with LLM: {type(self.llm_model).__name__}")

        # Advanced query engine for complex queries
        self.advanced_query_engine = SQLTableRetrieverQueryEngine(
            sql_database=sql_database,
            table_retriever=self.table_index.as_retriever(similarity_top_k=1),
            llm=self.llm_model,  # Now using the correct LlamaIndex LLM model
            text_to_sql_prompt=self.text_to_sql_prompt,
            response_synthesis_prompt=self.response_synthesis_prompt,
            refine_synthesis_prompt=self.refine_synthesis_prompt,
        )
        logger.debug(f"Initialized advanced query engine with {len(schema_nodes)} schema nodes")

        logger.info("Initialized query engines for tables: %s", ", ".join(self.tables))

    def _execute_duckdb_query(self, sql_query: str) -> pd.DataFrame:
        """Execute a SQL query against DuckDB.

        This function is passed to LlamaIndex for SQL execution.

        Args:
            sql_query: SQL query to execute

        Returns:
            pandas DataFrame with results
        """
        return self.execute_query(sql_query)

    def load_dataframe(
        self, df: pd.DataFrame | list[pd.DataFrame], table_name: str | list[str]
    ) -> bool:
        """Load DataFrame(s) into the database and initialize query engines.

        Args:
            df: DataFrame or list of DataFrames to load
            table_name: Name of the table to create or list of table names

        Returns:
            bool: True if successful
        """
        # Use the base class method to load the DataFrame(s)
        success = super().load_dataframe(df, table_name)

        # If successful, initialize the query engines
        if success:
            self.initialize()

        return success

    def process_query(
        self,
        query: str,
        query_type: Literal["sql", "natural_language"],
        complexity: Literal["simple", "advanced"] = "simple",
    ) -> dict[str, Any]:
        """Process SQL or natural language queries with conversation memory.

        Args:
            query: The query string to process
            query_type: Either "sql" or "natural_language"
            complexity: For natural language queries, use "simple" for single table
                        queries or "advanced" for multi-table or complex queries

        Returns:
            Dict with query results and metadata
        """
        logger.debug(f"PROCESS_QUERY start: self.tables = {self.tables}")
        result = {"success": False}

        try:
            if not self.table_schemas:
                self.initialize()

            # Store user message in memory - using correct API
            from llama_index.core.memory.chat_memory_buffer import ChatMessage

            self.memory.put(ChatMessage(role="user", content=query))

            if query_type == "sql" and "JOIN" in query.upper():
                # Add optimization hints for complex joins
                query = f"PRAGMA enable_optimizer; {query}"

            if query_type == "sql":
                # Direct SQL execution
                result_df = self.execute_query(query)
                response = f"SQL query executed. {len(result_df)} rows returned."
                # Store assistant response in memory
                self.memory.put(ChatMessage(role="assistant", content=response))
                result["success"] = True
                result["data"] = result_df
                result["query_type"] = "sql"
            else:
                # Process natural language query
                if not self.simple_query_engine:
                    error_msg = "LlamaIndex query engines not initialized. Please load data first."
                    # Store assistant response in memory
                    self.memory.put(ChatMessage(role="assistant", content=error_msg))
                    result["error"] = error_msg
                    result["query_type"] = "natural_language"
                else:
                    # Choose the appropriate engine based on complexity
                    engine = (
                        self.advanced_query_engine
                        if complexity == "advanced" and self.advanced_query_engine
                        else self.simple_query_engine
                    )

                    # Include chat history in the query context
                    chat_history = self.get_chat_history()

                    # Modify query to include context if we have chat history
                    if chat_history:
                        # Prepend the chat history to the query for context
                        contextualized_query = f"Given the following conversation history:\n{chat_history}\n\nNew question: {query}"
                    else:
                        contextualized_query = query

                    logger.debug(f"Sending query to engine: {contextualized_query[:100]}...")
                    # Execute the query without the context parameter
                    response = engine.query(contextualized_query)
                    logger.debug(f"Generated SQL: {response.metadata.get('sql_query', 'none')}")

                    # Store assistant response in memory
                    self.memory.put(ChatMessage(role="assistant", content=response.response))

                    result["success"] = True
                    result["data"] = response.response
                    result["sql_query"] = response.metadata["sql_query"]
                    result["raw_data"] = response.metadata.get("result")
                    result["explanation"] = f"Converted natural language to SQL: {response.metadata['sql_query']}"
                    result["query_type"] = "natural_language"

        except Exception as e:
            logger.exception("Error processing query")
            result["error"] = str(e)

        logger.debug(f"PROCESS_QUERY end: self.tables = {self.tables}")
        return result

    def get_chat_history(self) -> str:
        """Get the formatted chat history for context.

        Returns:
            Formatted string of chat history
        """
        messages = self.memory.get_all()
        if not messages:
            return ""

        history = []
        for message in messages:
            prefix = "User: " if message.role == "user" else "Assistant: "
            history.append(f"{prefix}{message.content}")

        return "\n".join(history)

    def clear_chat_history(self) -> None:
        """Clear the chat history but keep the chat store."""
        self.memory.reset()
        logger.info("Chat history cleared")

    def change_chat_session(self, new_session_key: str) -> None:
        """Switch to a different chat session.

        Args:
            new_session_key: New session identifier
        """
        current_store = self.memory.chat_store
        memory_type = "summary" if isinstance(self.memory, ChatSummaryMemoryBuffer) else "simple"
        token_limit = self.memory.token_limit

        self._init_chat_memory(memory_type, token_limit, current_store, new_session_key)
        logger.info(f"Switched to chat session: {new_session_key}")

    def clear_data(self) -> bool:
        """Clear all data and reset LlamaIndex components.

        Returns:
            bool: True if successful
        """
        # Use the base class method to clear data
        success = super().clear_data()

        # Reset LlamaIndex components
        if success:
            self.table_schemas = {}
            self.simple_query_engine = None
            self.advanced_query_engine = None
            self.table_index = None
            # Don't clear chat memory by default - that's a separate operation

        return success

    def get_tables(self) -> list[str]:
        """Get a list of all tables in the DuckDB database.

        Returns:
            List of table names as strings
        """
        return self.tables

    def __del__(self):
        """Clean up database connections."""
        # Call parent destructor
        super().__del__()


# # Demo usage if run directly
# if __name__ == "__main__":
#     # Code for testing
#     from llama_index.llms.openai import ChatOpenAI
#     from llama_index.embeddings.openai import OpenAIEmbedding
#
#     llm = ChatOpenAI(model="gpt-3.5-turbo")
#     embed = OpenAIEmbedding()
#     
#     db_service = EnhancedDuckDBService(
#         embed_model=embed,
#         llm_model=llm
#     )
#
#     # Test queries
#     output1 = db_service.process_query("What tables do you have?", "natural_language")
#     output2 = db_service.process_query("How many products are there?", "natural_language")
#     output3 = db_service.process_query("What did I ask before?", "natural_language")
#
#     print(output1["data"])
#     print(output2["data"])
#     print(output3["data"])

