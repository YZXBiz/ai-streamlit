"""
Data chat agent implementation for processing natural language queries.

This module provides functions for converting natural language queries into SQL,
executing the SQL, and analyzing the results.
"""

from typing import Any

from assortment_chatbot.services.duckdb_service import DuckDBService
from assortment_chatbot.utils.logging import get_logger

logger = get_logger(__name__)


def process_query(
    query: str,
    db_service: DuckDBService,
    table_name: str,
    context: dict | None = None,
) -> dict[str, Any]:
    """
    Process a natural language query and generate SQL.

    Args:
        query: The user's natural language query
        db_service: The DuckDB service to use for schema information
        table_name: The name of the table to query
        context: Optional context information including schema details, conversation history, etc.

    Returns:
        Dict with SQL query and explanation
    """
    # This is a placeholder implementation
    # In a real implementation, this would use PydanticAI to convert natural language to SQL with context

    # Get schema information
    context.get("schema_info") if context else db_service.get_schema_info()

    # Use conversation history if available in context
    if context and "conversation_history" in context:
        context["conversation_history"]

    # Simple logic to generate a more contextual SQL based on the query
    if "count" in query.lower() or "how many" in query.lower():
        return {
            "sql": f"SELECT COUNT(*) FROM {table_name}",
            "explanation": f"This query counts all rows in the {table_name} table.",
        }
    elif "average" in query.lower() or "avg" in query.lower() or "mean" in query.lower():
        # Try to extract column name after "average of"
        parts = query.lower().split("average of ")
        if len(parts) > 1:
            col_candidate = parts[1].split()[0]
            # This is simplistic - in a real implementation, we'd validate this column exists
            return {
                "sql": f"SELECT AVG({col_candidate}) FROM {table_name}",
                "explanation": f"This query calculates the average of the {col_candidate} column in the {table_name} table.",
            }

    # For now, just return a simple SQL query based on the table name
    return {
        "sql": f"SELECT * FROM {table_name} LIMIT 10",
        "explanation": f"This query selects all columns from the {table_name} table, limited to 10 rows.",
    }


def validate_and_refine_sql(
    sql_query: str, db_service: DuckDBService, table_name: str
) -> dict[str, Any]:
    """
    Validate the generated SQL and refine it if needed.

    Args:
        sql_query: The SQL query to validate
        db_service: The DuckDB service to use for schema validation
        table_name: The name of the main table

    Returns:
        Dict with validation results and possibly refined SQL
    """
    # Get schema information
    schema_info = db_service.get_schema_info()

    # Extract table columns if available
    table_columns = []
    if "columns" in schema_info and table_name in schema_info["columns"]:
        table_columns = schema_info["columns"][table_name]

    # Initialize result
    result = {
        "is_valid": True,
        "issues": [],
    }

    # Check for common SQL issues
    if "FROM" not in sql_query.upper():
        result["is_valid"] = False
        result["issues"].append("Missing FROM clause")
        result["refined_sql"] = f"SELECT * FROM {table_name} LIMIT 10"
        return result

    # Check for proper table name
    if table_name not in sql_query:
        result["is_valid"] = False
        result["issues"].append(f"Query doesn't reference table {table_name}")

        # Try to fix by replacing other table references with correct table
        parts = sql_query.upper().split("FROM")
        if len(parts) > 1:
            # Extract the part after FROM and before the next clause
            from_part = parts[1].strip().split(" ")[0]
            refined_sql = sql_query.replace(from_part, table_name)
            result["refined_sql"] = refined_sql
        else:
            result["refined_sql"] = f"SELECT * FROM {table_name} LIMIT 10"

        return result

    # Check for invalid column references if we have column information
    if table_columns:
        # This is a simplistic check - real implementation would be more sophisticated
        select_part = sql_query.upper().split("FROM")[0].replace("SELECT", "").strip()

        # If SELECT *, no need to check columns
        if select_part != "*":
            # Check if any requested columns don't exist in the table
            for col in select_part.split(","):
                col = col.strip()
                if col not in table_columns and col != "*" and not col.startswith("COUNT("):
                    result["is_valid"] = False
                    result["issues"].append(f"Column '{col}' not found in table {table_name}")

                    # Replace the invalid column with a valid one or use *
                    result["refined_sql"] = f"SELECT * FROM {table_name} LIMIT 10"
                    return result

    # If valid, don't provide refined SQL
    return result


def execute_query_and_analyze(sql_query: str, db_service: DuckDBService) -> dict[str, Any]:
    """
    Execute a SQL query and analyze the results.

    Args:
        sql_query: The SQL query to execute
        db_service: The DuckDB service to use for execution

    Returns:
        Dict with results and statistics
    """
    try:
        # Execute the query
        result_df = db_service.execute_query(sql_query)

        # If execution was successful, analyze the results
        if result_df is not None:
            # Get basic statistics
            row_count = len(result_df)
            column_count = len(result_df.columns)

            return {
                "success": True,
                "result_data": result_df,
                "stats": {
                    "row_count": row_count,
                    "column_count": column_count,
                },
            }
        else:
            return {
                "success": False,
                "error": "Query execution failed with no specific error message.",
            }
    except Exception as e:
        logger.error("Error executing query", exc_info=True)
        return {
            "success": False,
            "error": f"Error executing query: {str(e)}",
        }
