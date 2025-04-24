"""
Enhanced DuckDB service with LlamaIndex integration.

This module provides an enhanced DuckDB service that combines traditional
SQL capabilities with natural language query processing using LlamaIndex.
"""

from typing import Any, Literal

import duckdb
import pandas as pd
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

# Update import for flat structure
from logger import get_logger
from sqlalchemy import create_engine

logger = get_logger(__name__)


class DuckDBService:
    """Base DuckDB service for data storage and querying."""

    def __init__(self, db_path: str | None = None):
        """Initialize the DuckDB service.

        Args:
            db_path: Optional path to DuckDB database file. If None, an in-memory
                    database will be used.
        """
        self.conn = duckdb.connect(database=db_path if db_path else ":memory:")
        self.tables: list[str] = []

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a SQL query.

        Args:
            query: SQL query to execute

        Returns:
            DataFrame with results
        """
        result = self.conn.execute(query).fetchdf()
        return result

    def load_dataframe(
        self, df: pd.DataFrame | list[pd.DataFrame], table_name: str | list[str]
    ) -> bool:
        """Load a DataFrame or list of DataFrames into DuckDB with transaction support.

        Args:
            df: DataFrame or list of DataFrames to load
            table_name: Name of the table to create or list of table names

        Returns:
            bool: True if successful
        """
        # Handle single DataFrame case
        if isinstance(df, pd.DataFrame) and isinstance(table_name, str):
            return self._load_single_dataframe(df, table_name)

        # Handle list of DataFrames case
        elif isinstance(df, list) and isinstance(table_name, list):
            if len(df) != len(table_name):
                logger.error("Number of DataFrames and table names must match")
                return False

            try:
                self.conn.execute("BEGIN TRANSACTION")
                for single_df, single_table in zip(df, table_name, strict=True):
                    if not isinstance(single_df, pd.DataFrame):
                        raise TypeError(f"Expected DataFrame, got {type(single_df)}")
                    self.conn.register(single_table, single_df)
                    self.tables.append(single_table)
                self.conn.execute("COMMIT")
                return True
            except Exception as e:
                self.conn.execute("ROLLBACK")
                logger.error("Failed to load DataFrames: %s", e)
                return False
        else:
            logger.error(
                "Invalid parameter types. Both df and table_name must be either single values or lists."
            )
            return False

    def _load_single_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Load a single DataFrame into DuckDB.

        Args:
            df: DataFrame to load
            table_name: Name of the table to create

        Returns:
            bool: True if successful
        """
        try:
            self.conn.execute("BEGIN TRANSACTION")
            self.conn.register(table_name, df)
            self.tables.append(table_name)
            self.conn.execute("COMMIT")
            return True
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error("Failed to load DataFrame: %s", e)
            return False

    def get_schema_info(self) -> dict[str, Any]:
        """Get schema information for all tables.

        Returns:
            Dict with schema information
        """
        schema_info = {
            "tables": self.tables,
            "columns": {},
        }

        for table in self.tables:
            # Get column information
            columns = self.execute_query(f"DESCRIBE {table}")
            schema_info["columns"][table] = columns["column_name"].tolist()

        return schema_info

    def clear_data(self) -> bool:
        """Clear all data from the service.

        Returns:
            bool: True if successful
        """
        try:
            # Drop all tables and views
            for table in self.tables:
                try:
                    # Try to drop as view
                    self.conn.execute(f"DROP VIEW IF EXISTS {table}")
                except Exception as e:
                    # Ignore catalog/type mismatch errors
                    if "is of type Table, trying to drop type View" not in str(e):
                        logger.warning(f"Error dropping view {table}: {e}")

                try:
                    # Try to drop as table
                    self.conn.execute(f"DROP TABLE IF EXISTS {table}")
                except Exception as e:
                    # Ignore catalog/type mismatch errors
                    if "is of type View, trying to drop type Table" not in str(e):
                        logger.warning(f"Error dropping table {table}: {e}")

            self.tables = []
            return True
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False

    def __del__(self):
        """Clean up database connection."""
        self.conn.close()

    def load_file_directly(self, file_path: str, table_name: str) -> bool:
        """Load a file directly into DuckDB using native loaders.

        Bypasses pandas DataFrame conversion to avoid Arrow compatibility issues.

        Args:
            file_path: Path to the file to load
            table_name: Name of the table to create

        Returns:
            bool: True if successful
        """
        try:
            file_extension = file_path.split(".")[-1].lower()
            self.conn.execute("BEGIN TRANSACTION")

            if file_extension == "csv":
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_path}')"
                )
            elif file_extension == "parquet":
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{file_path}')"
                )
            elif file_extension in ["xls", "xlsx"]:
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_excel('{file_path}')"
                )
            elif file_extension == "json":
                self.conn.execute(
                    f"CREATE TABLE {table_name} AS SELECT * FROM read_json_auto('{file_path}')"
                )
            else:
                logger.error("Unsupported file extension: %s", file_extension)
                self.conn.execute("ROLLBACK")
                return False

            # Add to tables list if successful
            self.tables.append(table_name)
            self.conn.execute("COMMIT")
            logger.info("Successfully loaded %s into table %s", file_path, table_name)
            return True

        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error("Failed to load file directly: %s", e)
            return False


class EnhancedDuckDBService(DuckDBService):
    """Enhanced DuckDB service with LlamaIndex integration for NL2SQL.

    This class extends the base DuckDBService with LlamaIndex capabilities
    to support natural language to SQL conversion, query retrieval based on
    schema understanding, and contextual responses.
    """

    def __init__(
        self,
        embed_model: OpenAIEmbedding,
        db_path: str | None = None,
        memory_type: Literal["simple", "summary"] = "simple",
        token_limit: int = 3000,
        chat_store: BaseChatStore | None = None,
        chat_store_key: str = "default_session",
    ):
        """Initialize the enhanced DuckDB service.

        Args:
            embed_model: OpenAI embedding model for LlamaIndex
            db_path: Optional path to DuckDB database file. If None, an in-memory
                    database will be used.
            memory_type: Type of chat memory to use ("simple" or "summary")
            token_limit: Maximum number of tokens to store in memory
            chat_store: Optional custom chat store for persistence
            chat_store_key: Key to identify this chat session
        """
        # Initialize the base DuckDBService
        super().__init__(db_path)

        # Initialize LlamaIndex components
        self.table_schemas = {}
        self.simple_query_engine = None
        self.advanced_query_engine = None
        self.table_index = None
        self.embed_model = embed_model

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
        # Get table schemas directly from DuckDB
        self._get_table_schemas()

        # Initialize query engines if we have tables
        if self.tables:
            self._init_query_engines()
            logger.info("Initialized LlamaIndex query engines")
        else:
            logger.info("No tables available for LlamaIndex initialization")

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

        # Create a SQLAlchemy engine for DuckDB
        # Use a unique connection string with a timestamp to avoid conflicts
        import time

        timestamp = int(time.time())
        temp_db_path = f":memory:{timestamp}"
        engine = create_engine(f"duckdb:///{temp_db_path}")

        # Create tables in the SQLAlchemy engine
        for table_name in self.tables:
            df = self.execute_query(f"SELECT * FROM {table_name}")
            df.to_sql(table_name, engine, index=False, if_exists="replace")

        # Initialize SQLDatabase with the SQLAlchemy engine
        sql_database = SQLDatabase(engine=engine, include_tables=list(self.table_schemas.keys()))

        # Initialize simple query engine for natural language queries
        self.simple_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=list(self.table_schemas.keys()),
        )

        # Advanced query engine for complex queries
        self.advanced_query_engine = SQLTableRetrieverQueryEngine(
            sql_database=sql_database,
            table_retriever=self.table_index.as_retriever(similarity_top_k=1),
        )

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
            return {"success": True, "data": result_df, "query_type": "sql"}
        else:
            # Process natural language query
            if not self.simple_query_engine:
                error_msg = "LlamaIndex query engines not initialized. Please load data first."
                # Store assistant response in memory
                self.memory.put(ChatMessage(role="assistant", content=error_msg))
                return {
                    "success": False,
                    "error": error_msg,
                    "query_type": "natural_language",
                }

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

            # Execute the query without the context parameter
            response = engine.query(contextualized_query)

            # Store assistant response in memory
            self.memory.put(ChatMessage(role="assistant", content=response.response))

            return {
                "success": True,
                "data": response.response,
                "sql_query": response.metadata["sql_query"],
                "raw_data": response.metadata.get("result"),
                "explanation": f"Converted natural language to SQL: {response.metadata['sql_query']}",
                "query_type": "natural_language",
            }

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

    def __del__(self):
        """Clean up database connections."""
        # Call parent destructor
        super().__del__()


if __name__ == "__main__":
    import pandas as pd
    from llama_index.embeddings.openai import OpenAIEmbedding
    from settings import SETTINGS

    # Initialize with memory options
    db_service = EnhancedDuckDBService(
        embed_model=OpenAIEmbedding(api_key=SETTINGS.OPENAI_API_KEY),
        memory_type="summary",  # Use summarizing memory for longer conversations
        token_limit=4000,  # Adjust token limit as needed
        chat_store_key="user_123",  # Track sessions by user ID
    )

    df = pd.DataFrame({"name": ["John", "Jane", "Jim"], "age": [25, 30, 35]})
    db_service.load_dataframe(df, "test")

    # First query
    output1 = db_service.process_query("top 5 records", "natural_language")

    # Follow-up query (will use conversation context)
    output2 = db_service.process_query("Who is older than that?", "natural_language")

    #
    output3 = db_service.process_query("What did I ask before?", "natural_language")

    output1
    print(output2["data"])
    print(output3["data"])
