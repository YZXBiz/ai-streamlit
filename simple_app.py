"""
Simple Data Chat Assistant - A streamlined, single-file implementation

This file contains all essential functionality for a data chatbot in one place:
- Data loading and processing
- Chat interface
- DuckDB integration
- Natural language query processing

Run with: streamlit run simple_app.py
"""

import os
import tempfile
import traceback
import uuid
from typing import Any, Literal

import duckdb
import pandas as pd
import streamlit as st

# Import LlamaIndex components directly
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.struct_store import (
    NLSQLTableQueryEngine,
    SQLTableRetrieverQueryEngine,
)
from llama_index.core.indices.struct_store.sql_query import SQLDatabase
from llama_index.core.schema import Node
from llama_index.embeddings.openai import OpenAIEmbedding
from sqlalchemy import create_engine

# App configuration
APP_TITLE = "Simple Data Chat Assistant"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Talk to your data using natural language"

# Set page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS
st.markdown(
    """
<style>
    .main .block-container {padding-top: 2rem;}
    .stApp {max-width: 1200px; margin: 0 auto;}
    h1 {margin-bottom: 0.5rem !important;}
    .stTabs [data-baseweb="tab-panel"] {padding-top: 1rem;}
</style>
""",
    unsafe_allow_html=True,
)

# Initialize OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Make sure OpenAI API key is provided
if not OPENAI_API_KEY:
    st.error("⚠️ No OpenAI API key found! Please set the OPENAI_API_KEY environment variable.")
    st.stop()


# Create a simple logger using Streamlit
def log(level, message):
    """Simple logging to Streamlit's debug area"""
    if level == "error":
        print(f"ERROR: {message}")
    elif level == "warning":
        print(f"WARNING: {message}")
    else:
        print(f"INFO: {message}")


# DuckDB Service integrated directly
class DuckDBService:
    """Simple DuckDB database service with LlamaIndex integration"""

    def __init__(self):
        """Initialize the database service"""
        self.conn = duckdb.connect(database=":memory:")
        self.tables = []
        self.table_schemas = {}
        self.simple_query_engine = None
        self.advanced_query_engine = None
        self.table_index = None

        # Initialize the embedding model
        self.embed_model = OpenAIEmbedding(
            api_key=OPENAI_API_KEY,
            timeout=45.0,
            max_retries=3,
        )

    def load_file(self, file_path, table_name):
        """Load a file directly into DuckDB"""
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
                log("error", f"Unsupported file extension: {file_extension}")
                self.conn.execute("ROLLBACK")
                return False

            # Add to tables list if successful
            self.tables.append(table_name)
            self.conn.execute("COMMIT")
            log("info", f"Successfully loaded {file_path} into table {table_name}")

            # Initialize query engines
            self.initialize()
            return True

        except Exception as e:
            self.conn.execute("ROLLBACK")
            log("error", f"Failed to load file: {e}")
            return False

    def load_dataframe(self, df, table_name):
        """Load a DataFrame into DuckDB"""
        try:
            self.conn.execute("BEGIN TRANSACTION")
            self.conn.register(table_name, df)
            self.tables.append(table_name)
            self.conn.execute("COMMIT")

            # Initialize query engines
            self.initialize()
            return True
        except Exception as e:
            self.conn.execute("ROLLBACK")
            log("error", f"Failed to load DataFrame: {e}")
            return False

    def execute_query(self, query):
        """Execute a SQL query"""
        return self.conn.execute(query).fetchdf()

    def initialize(self):
        """Initialize LlamaIndex components"""
        # Get table schemas
        self._get_table_schemas()

        # Initialize query engines if we have tables
        if self.tables:
            self._init_query_engines()

    def _get_table_schemas(self):
        """Get schema information for all tables"""
        self.table_schemas = {}

        for table in self.tables:
            # Get column information
            columns_df = self.execute_query(f"DESCRIBE {table}")
            # Store schema information
            column_names = columns_df["column_name"].tolist()
            column_types = columns_df["column_type"].tolist()
            self.table_schemas[table] = {
                "name": table,
                "columns": column_names,
                "types": column_types,
            }

    def _init_query_engines(self):
        """Initialize LlamaIndex query engines"""
        if not self.table_schemas:
            return

        # Create schema nodes for vector search
        schema_nodes = []
        for table_name, schema_info in self.table_schemas.items():
            # Create a detailed description of the table for embedding
            schema_text = (
                f"Table '{table_name}' with description: collection of {table_name} data. "
                f"Columns: {', '.join([f'{col} ({col_type})' for col, col_type in zip(schema_info['columns'], schema_info['types'])])}"
            )
            # Create Node with text directly
            node = Node(text=schema_text, metadata={"table_name": table_name})
            schema_nodes.append(node)

        # Create vector index from schema nodes
        self.table_index = VectorStoreIndex(schema_nodes, embed_model=self.embed_model)

        # Create a SQLAlchemy engine for DuckDB
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

        # Initialize query engines
        self.simple_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=list(self.table_schemas.keys()),
        )

        self.advanced_query_engine = SQLTableRetrieverQueryEngine(
            sql_database=sql_database,
            table_retriever=self.table_index.as_retriever(similarity_top_k=1),
        )

    def process_query(self, query, query_type="natural_language", complexity="simple"):
        """Process SQL or natural language queries"""
        if not self.table_schemas:
            self.initialize()

        if query_type == "sql":
            # Direct SQL execution
            result_df = self.execute_query(query)
            return {"success": True, "data": result_df, "query_type": "sql"}
        else:
            # Process natural language query
            if not self.simple_query_engine:
                return {
                    "success": False,
                    "error": "Query engines not initialized. Please load data first.",
                    "query_type": "natural_language",
                }

            # Choose the appropriate engine based on complexity
            engine = (
                self.advanced_query_engine
                if complexity == "advanced" and self.advanced_query_engine
                else self.simple_query_engine
            )

            try:
                # Execute the query
                response = engine.query(query)

                return {
                    "success": True,
                    "data": response.response,
                    "sql_query": response.metadata["sql_query"],
                    "raw_data": response.metadata.get("result"),
                    "explanation": f"Converted natural language to SQL: {response.metadata['sql_query']}",
                    "query_type": "natural_language",
                }
            except Exception as e:
                log("error", f"Error processing query: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "query_type": "natural_language",
                }


# Create a function to save uploaded files
def save_uploaded_file(uploaded_file):
    """Save an uploaded file to disk in a temp directory"""
    # Create a data directory if it doesn't exist
    data_dir = os.path.join("data", "temp")
    os.makedirs(data_dir, exist_ok=True)

    # Generate a unique filename to avoid collisions
    file_extension = uploaded_file.name.split(".")[-1].lower()
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{unique_id}_{uploaded_file.name}"
    file_path = os.path.join(data_dir, filename)

    # Save the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    log("info", f"Saved uploaded file to {file_path}")
    return file_path


# Initialize the DB service directly in the global scope
db_service = DuckDBService()


# Define the main app structure with simple tabs
def main():
    """Run the streamlined data chat application"""

    # Navigation with simple tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "📤 Upload Data", "💬 Chat"])

    # === HOME TAB ===
    with tab1:
        st.title("Simple Data Chat Assistant")
        st.write("""
        Welcome to Simple Data Chat Assistant! This tool allows you to:
        
        1. Upload your data files (CSV, Excel, JSON, Parquet)
        2. Ask questions about your data in plain English
        3. Get instant insights with AI-powered analysis
        
        To get started, go to the **Upload Data** tab and upload your file.
        """)

        st.info(f"Version: {APP_VERSION}")

    # === UPLOAD DATA TAB ===
    with tab2:
        st.title("📤 Upload Data")

        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=["csv", "xlsx", "xls", "json", "parquet"],
            help="Maximum file size: 200MB",
        )

        if uploaded_file is not None:
            # Save the file to disk
            file_path = save_uploaded_file(uploaded_file)

            # Store path in session state
            st.session_state.file_path = file_path
            st.session_state.file_name = uploaded_file.name

            # Load file into DB directly based on extension
            table_name = uploaded_file.name.split(".")[0].lower().replace(" ", "_")

            # Try to load the file
            success = db_service.load_file(file_path, table_name)

            if success:
                st.success(f"Successfully loaded {uploaded_file.name}")

                # Preview the data
                preview_df = db_service.execute_query(f"SELECT * FROM {table_name} LIMIT 10")

                st.subheader("Data Preview")
                st.dataframe(preview_df, use_container_width=True)

                # Show table info
                row_count = db_service.execute_query(
                    f"SELECT COUNT(*) AS count FROM {table_name}"
                ).iloc[0, 0]
                st.info(f"Loaded {row_count} rows and {len(preview_df.columns)} columns")

                # Store in session for other tabs
                st.session_state.data_loaded = True
            else:
                st.error("Failed to load the file. Please check the file format.")

    # === CHAT TAB ===
    with tab3:
        st.title("💬 Chat with Your Data")

        # Check if data is loaded
        if "data_loaded" not in st.session_state or not st.session_state.data_loaded:
            st.warning("Please upload data first in the Upload Data tab.")
            return

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Handle user input
        if prompt := st.chat_input("Ask a question about your data..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    # Determine complexity based on simple heuristic
                    complexity = "advanced" if len(db_service.tables) > 1 else "simple"

                    try:
                        # Process the query
                        result = db_service.process_query(
                            query=prompt, query_type="natural_language", complexity=complexity
                        )

                        if result["success"]:
                            response_text = result.get("data", "No response found.")
                            sql_query = result.get("sql_query", "SQL query not available.")

                            # Format response
                            formatted_response = (
                                f"**Analysis Result:**\n{response_text}\n\n---\n"
                                f"*Executed SQL Query:*\n```sql\n{sql_query}\n```"
                            )

                            # Display formatted response
                            st.markdown(formatted_response)

                            # Add to chat history
                            st.session_state.messages.append(
                                {"role": "assistant", "content": formatted_response}
                            )
                        else:
                            error_message = result.get("error", "Unknown error")
                            st.error(f"Error processing query: {error_message}")

                            # Add error to chat history
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"❌ Error: {error_message}"}
                            )

                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"❌ Error: {str(e)}"}
                        )

        # Add clear chat button
        if st.button("Clear Chat") and len(st.session_state.messages) > 0:
            st.session_state.messages = []
            st.rerun()


# Run the app
if __name__ == "__main__":
    main()
