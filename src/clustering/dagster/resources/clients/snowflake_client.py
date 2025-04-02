"""Snowflake client resource for Dagster."""

import os
import time
from typing import Any, Dict, List

import dagster as dg
import snowflake.connector  # type: ignore
from pydantic import BaseModel, Field, SecretStr
from snowflake.connector.errors import DatabaseError, OperationalError, ProgrammingError  # type: ignore


class SnowflakeClientSchema(BaseModel):
    """Schema for Snowflake client configuration."""

    account: str = Field(..., description="Snowflake account identifier")
    user: str = Field(..., description="Snowflake username")
    password: SecretStr = Field(..., description="Snowflake password")
    warehouse: str = Field(..., description="Snowflake warehouse name")
    database: str = Field(None, description="Default Snowflake database")
    schema: str = Field(None, description="Default Snowflake schema")
    role: str = Field(None, description="Snowflake role")
    authenticator: str = Field(None, description="Authentication method (e.g., 'externalbrowser')")
    timeout: int = Field(60, description="Connection timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retry attempts")


@dg.resource(config_schema=SnowflakeClientSchema.model_json_schema())
def snowflake_client(context: dg.InitResourceContext) -> "SnowflakeClientWrapper":
    """Resource for Snowflake client.

    Args:
        context: The Dagster resource initialization context

    Returns:
        SnowflakeClientWrapper: A wrapper around Snowflake connector with retry logic
    """
    # Get configuration values with environment variable fallback
    config = context.resource_config
    account = config.get("account", os.environ.get("SNOWFLAKE_ACCOUNT"))
    user = config.get("user", os.environ.get("SNOWFLAKE_USER"))
    password = config.get("password", os.environ.get("SNOWFLAKE_PASSWORD"))

    if password and isinstance(password, SecretStr):
        password = password.get_secret_value()
    else:
        password = os.environ.get("SNOWFLAKE_PASSWORD")

    warehouse = config.get("warehouse", os.environ.get("SNOWFLAKE_WAREHOUSE"))
    database = config.get("database", os.environ.get("SNOWFLAKE_DATABASE"))
    schema = config.get("schema", os.environ.get("SNOWFLAKE_SCHEMA"))
    role = config.get("role", os.environ.get("SNOWFLAKE_ROLE"))
    authenticator = config.get("authenticator", os.environ.get("SNOWFLAKE_AUTHENTICATOR"))
    timeout = config.get("timeout", 60)
    max_retries = config.get("max_retries", 3)

    # Configure connection parameters
    conn_params = {
        "account": account,
        "user": user,
        "password": password,
        "warehouse": warehouse,
        "timeout": timeout,
    }

    # Add optional parameters
    if database:
        conn_params["database"] = database
    if schema:
        conn_params["schema"] = schema
    if role:
        conn_params["role"] = role
    if authenticator:
        conn_params["authenticator"] = authenticator

    logger = context.log

    try:
        if logger:
            logger.info(f"Creating Snowflake connection to account {account}")

        # Don't create a connection here, as Snowflake best practice is to create
        # a fresh connection for each operation (handled by the wrapper)
        return SnowflakeClientWrapper(conn_params, max_retries, logger)

    except Exception as e:
        if logger:
            logger.error(f"Error initializing Snowflake client: {e}")
        raise


class SnowflakeClientWrapper:
    """Wrapper around Snowflake connector with retry logic."""

    def __init__(self, conn_params: Dict[str, Any], max_retries: int, logger):
        """Initialize the Snowflake client wrapper.

        Args:
            conn_params: Connection parameters for Snowflake
            max_retries: Maximum number of retry attempts
            logger: Logger object
        """
        self.conn_params = conn_params
        self.max_retries = max_retries
        self.logger = logger

    def _get_connection(self):
        """Create a new Snowflake connection."""
        return snowflake.connector.connect(**self.conn_params)

    def _with_retry(self, operation_name: str, operation_func, *args, **kwargs):
        """Execute an operation with retry logic.

        Args:
            operation_name: Name of the operation for logging
            operation_func: Function to execute
            *args, **kwargs: Arguments to pass to the operation function

        Returns:
            Result of the operation

        Raises:
            Exception: If the operation fails after max_retries attempts
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                conn = self._get_connection()
                try:
                    result = operation_func(conn, *args, **kwargs)
                    return result
                finally:
                    conn.close()

            except (ProgrammingError, DatabaseError, OperationalError) as e:
                last_exception = e
                if self.logger:
                    self.logger.warning(
                        f"Snowflake {operation_name} attempt {attempt+1}/{self.max_retries} "
                        f"failed with error: {str(e)}"
                    )

                # Skip retry for certain error codes
                if (
                    hasattr(e, "errno")
                    and e.errno
                    in [
                        # Add specific error codes that shouldn't be retried
                        # e.g., 1234  # Invalid object name
                    ]
                ):
                    if self.logger:
                        self.logger.error(f"Non-retryable Snowflake error: {str(e)}")
                    raise

                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = 2**attempt + (0.1 * attempt)
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(f"Snowflake {operation_name} failed after {self.max_retries} attempts")

        if last_exception:
            raise last_exception

    def execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a SQL query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of dictionaries containing query results
        """

        def _execute(conn, query, params):
            cursor = conn.cursor(snowflake.connector.DictCursor)
            try:
                cursor.execute(query, params or {})
                return cursor.fetchall()
            finally:
                cursor.close()

        return self._with_retry("execute_query", _execute, query, params)

    def execute_statement(self, statement: str, params: Dict = None) -> int:
        """Execute a SQL statement (INSERT, UPDATE, DELETE, etc.) and return row count.

        Args:
            statement: SQL statement to execute
            params: Statement parameters

        Returns:
            Number of rows affected
        """

        def _execute(conn, statement, params):
            cursor = conn.cursor()
            try:
                cursor.execute(statement, params or {})
                return cursor.rowcount
            finally:
                cursor.close()

        return self._with_retry("execute_statement", _execute, statement, params)

    def bulk_insert(self, table_name: str, columns: List[str], rows: List[List]) -> int:
        """Insert multiple rows into a table.

        Args:
            table_name: Target table name
            columns: Column names
            rows: List of row values to insert

        Returns:
            Number of rows inserted
        """

        def _bulk_insert(conn, table_name, columns, rows):
            column_str = ", ".join(columns)
            placeholders = ", ".join(["%s"] * len(columns))
            query = f"INSERT INTO {table_name} ({column_str}) VALUES ({placeholders})"

            cursor = conn.cursor()
            try:
                cursor.executemany(query, rows)
                return cursor.rowcount
            finally:
                cursor.close()

        return self._with_retry("bulk_insert", _bulk_insert, table_name, columns, rows)

    def copy_into_table(self, table_name: str, stage_path: str, file_format: str = "CSV", options: Dict = None) -> Dict:
        """Load data into a table using COPY INTO command.

        Args:
            table_name: Target table name
            stage_path: Snowflake stage path where data files are located
            file_format: File format (CSV, JSON, etc.)
            options: Additional COPY options

        Returns:
            Copy operation result details
        """

        def _copy_into(conn, table_name, stage_path, file_format, options):
            opts = []
            if options:
                for k, v in options.items():
                    opts.append(f"{k} = {v}")

            options_str = "\n".join(opts)
            query = f"""
            COPY INTO {table_name} 
            FROM {stage_path}
            FILE_FORMAT = (TYPE = '{file_format}')
            {options_str}
            """

            cursor = conn.cursor(snowflake.connector.DictCursor)
            try:
                cursor.execute(query)
                return cursor.fetchall()
            finally:
                cursor.close()

        return self._with_retry("copy_into_table", _copy_into, table_name, stage_path, file_format, options)
