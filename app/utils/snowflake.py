"""
Snowflake connector utility.

This module handles connections to Snowflake and query execution.
"""
import os
import pandas as pd
from typing import Dict, Any, Optional

from app.utils.logging import get_logger, log_operation, mdc_context
from app.utils.errors import ExternalAPIError, InvalidConfigurationError

# Get module logger
logger = get_logger(__name__)

try:
    import snowflake.connector
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.asymmetric import dsa
    from cryptography.hazmat.primitives import serialization
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    logger.warning("Snowflake connector dependencies not available")


class SnowflakeConnector:
    """Utility for connecting to Snowflake and executing queries."""
    
    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize Snowflake connector with credentials.
        
        Args:
            credentials: Dictionary with Snowflake connection parameters
                - account: Snowflake account identifier
                - user: Username
                - password: Password (or private_key_path)
                - warehouse: Warehouse name
                - database: Database name
                - schema: Schema name
                - role: User role (optional)
        """
        if not SNOWFLAKE_AVAILABLE:
            raise ImportError(
                "Snowflake connector dependencies not available. "
                "Please install with: uv add snowflake-connector-python cryptography"
            )
        
        self.credentials = credentials
        self.conn = None
        
        # Validate required credentials
        self._validate_credentials()
        
        # Set up MDC with sanitized connection info
        sanitized_creds = {
            "account": credentials.get("account"),
            "user": credentials.get("user"),
            "warehouse": credentials.get("warehouse"),
            "database": credentials.get("database"),
            "schema": credentials.get("schema"),
            "role": credentials.get("role"),
            # Omit password and private key info
        }
        self.connection_context = {"snowflake_connection": sanitized_creds}
    
    def _validate_credentials(self) -> None:
        """
        Validate that all required credentials are provided.
        
        Raises:
            InvalidConfigurationError: If required credentials are missing
        """
        required_fields = ["account", "user"]
        
        # Either password or private_key_path is required
        if "password" not in self.credentials and "private_key_path" not in self.credentials:
            required_fields.append("password/private_key_path")
        
        missing_fields = [field for field in required_fields if field not in self.credentials]
        
        if missing_fields:
            error_msg = f"Missing required Snowflake credentials: {', '.join(missing_fields)}"
            logger.error(error_msg)
            raise InvalidConfigurationError(error_msg)
    
    def connect(self) -> bool:
        """
        Establish connection to Snowflake.
        
        Returns:
            Success flag
            
        Raises:
            ExternalAPIError: On connection failure
        """
        with log_operation(logger, "snowflake_connect"):
            with mdc_context(**self.connection_context):
                try:
                    # Check for key-based authentication
                    if 'private_key_path' in self.credentials:
                        logger.info("Connecting to Snowflake using key-based authentication")
                        
                        with open(self.credentials['private_key_path'], 'rb') as key_file:
                            p_key = serialization.load_pem_private_key(
                                key_file.read(),
                                password=self.credentials.get('private_key_passphrase', None).encode() 
                                    if self.credentials.get('private_key_passphrase') else None,
                                backend=default_backend()
                            )
                        
                        # Get private key in PEM format
                        private_key = p_key.private_bytes(
                            encoding=serialization.Encoding.DER,
                            format=serialization.PrivateFormat.PKCS8,
                            encryption_algorithm=serialization.NoEncryption()
                        )
                        
                        self.conn = snowflake.connector.connect(
                            user=self.credentials['user'],
                            account=self.credentials['account'],
                            private_key=private_key,
                            warehouse=self.credentials.get('warehouse'),
                            database=self.credentials.get('database'),
                            schema=self.credentials.get('schema'),
                            role=self.credentials.get('role')
                        )
                    else:
                        # Password-based authentication
                        logger.info("Connecting to Snowflake using password authentication")
                        self.conn = snowflake.connector.connect(
                            user=self.credentials['user'],
                            password=self.credentials['password'],
                            account=self.credentials['account'],
                            warehouse=self.credentials.get('warehouse'),
                            database=self.credentials.get('database'),
                            schema=self.credentials.get('schema'),
                            role=self.credentials.get('role')
                        )
                    
                    logger.info("Successfully connected to Snowflake")
                    return True
                except Exception as e:
                    error_msg = f"Error connecting to Snowflake: {str(e)}"
                    logger.error(error_msg)
                    raise ExternalAPIError(
                        message=error_msg,
                        service_name="Snowflake",
                        details={"error_type": type(e).__name__}
                    ) from e
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame containing query results
            
        Raises:
            ExternalAPIError: On query execution failure
        """
        query_context = {
            **self.connection_context,
            "query_type": "SELECT" if query.strip().upper().startswith("SELECT") else "OTHER"
        }
        
        with log_operation(logger, "snowflake_query"):
            with mdc_context(**query_context):
                try:
                    if not self.conn:
                        self.connect()
                    
                    logger.info(f"Executing Snowflake query: {query[:100]}..." if len(query) > 100 else query)
                    cursor = self.conn.cursor()
                    cursor.execute(query)
                    
                    # Get column names
                    col_names = [col[0] for col in cursor.description]
                    
                    # Fetch data and convert to DataFrame
                    data = cursor.fetchall()
                    df = pd.DataFrame(data, columns=col_names)
                    
                    cursor.close()
                    logger.info(f"Query returned {len(df)} rows")
                    return df
                except Exception as e:
                    error_msg = f"Error executing Snowflake query: {str(e)}"
                    logger.error(error_msg)
                    raise ExternalAPIError(
                        message=error_msg,
                        service_name="Snowflake",
                        details={
                            "error_type": type(e).__name__,
                            "query_fragment": query[:200] if query else "None"
                        }
                    ) from e
    
    def get_table_schema(self, database: str, schema: str, table: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table.
        
        Args:
            database: Database name
            schema: Schema name
            table: Table name
            
        Returns:
            Dictionary with column names, types, and other metadata
            
        Raises:
            ExternalAPIError: On schema retrieval failure
        """
        schema_context = {
            **self.connection_context,
            "operation": "get_schema",
            "target_database": database,
            "target_schema": schema,
            "target_table": table
        }
        
        with log_operation(logger, "snowflake_get_schema"):
            with mdc_context(**schema_context):
                query = f"""
                SELECT 
                    column_name, 
                    data_type, 
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale,
                    is_nullable
                FROM 
                    {database}.information_schema.columns
                WHERE 
                    table_schema = '{schema}'
                    AND table_name = '{table}'
                ORDER BY 
                    ordinal_position
                """
                
                try:
                    schema_df = self.execute_query(query)
                    
                    if schema_df.empty:
                        error_msg = f"No schema information found for table {database}.{schema}.{table}"
                        logger.warning(error_msg)
                        return {"table_name": table, "columns": {}}
                    
                    # Convert to dictionary format
                    schema_info = {
                        "table_name": table,
                        "columns": {}
                    }
                    
                    for _, row in schema_df.iterrows():
                        schema_info["columns"][row['COLUMN_NAME']] = {
                            "data_type": row['DATA_TYPE'],
                            "max_length": row['CHARACTER_MAXIMUM_LENGTH'],
                            "precision": row['NUMERIC_PRECISION'],
                            "scale": row['NUMERIC_SCALE'],
                            "nullable": row['IS_NULLABLE'] == 'YES'
                        }
                    
                    logger.info(f"Retrieved schema for {database}.{schema}.{table} with {len(schema_info['columns'])} columns")
                    return schema_info
                except Exception as e:
                    error_msg = f"Error retrieving schema for {database}.{schema}.{table}: {str(e)}"
                    logger.error(error_msg)
                    
                    if isinstance(e, ExternalAPIError):
                        raise
                    
                    raise ExternalAPIError(
                        message=error_msg,
                        service_name="Snowflake",
                        details={
                            "error_type": type(e).__name__,
                            "database": database,
                            "schema": schema,
                            "table": table
                        }
                    ) from e
    
    def close(self) -> None:
        """Close the Snowflake connection."""
        if self.conn:
            logger.info("Closing Snowflake connection")
            self.conn.close()
            self.conn = None 