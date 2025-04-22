"""PydanticAI-based data chat agent for SQL generation and query analysis.

This module provides AI agent capabilities to process natural language queries,
generate SQL, and provide analysis of query results.
"""

import pandas as pd
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from dashboard.data.duckdb_service import DuckDBService
from dashboard.settings import get_settings

# Get settings
settings = get_settings()
agent_settings = settings.agent_settings

# Define data models for the agent

class SQLQuery(BaseModel):
    """Generated SQL query and metadata."""
    
    sql: str = Field(description="SQL query to execute against the database")
    explanation: str = Field(description="Explanation of what the SQL query does")
    visualization_type: Literal["table", "bar", "line", "scatter", "pie", "heatmap", "none"] = Field(
        description="Suggested visualization type for the result", 
        default="table"
    )
    transform_explanation: Optional[str] = Field(
        description="Explanation of how the data was transformed", 
        default=None
    )

@dataclass
class DBContext:
    """Database context for the agent."""
    db_service: DuckDBService
    current_table: str

# Create the PydanticAI agent for data chat
data_chat_agent = Agent(
    agent_settings["sql_agent_model"],  # Get model name from settings
    deps_type=DBContext,
    result_type=SQLQuery,
    system_prompt="""
    You are an expert SQL assistant that helps users explore and analyze their data.
    Your role is to:
    1. Understand the user's intent from their natural language query
    2. Generate appropriate SQL to fulfill the user's request
    3. Provide clear explanations of what the SQL does
    4. Suggest appropriate visualizations for the results
    
    Generate SQL based on their schema information and query.
    Be precise and efficient with your SQL. Focus on correctly answering the query.
    """,
    model_settings={"temperature": agent_settings["temperature"]}
)

# Add tools to enhance the agent's capabilities

@data_chat_agent.tool
def get_schema_info(ctx: RunContext[DBContext]) -> Dict[str, Any]:
    """Get schema information about the available tables and columns.
    
    Returns the database schema including table names and column details.
    """
    return ctx.deps.db_service.get_schema_info()

@data_chat_agent.tool
def get_table_preview(ctx: RunContext[DBContext], table_name: str, limit: int = None) -> Dict[str, Any]:
    """Get a preview of data in a specific table.
    
    Args:
        table_name: Name of the table to preview
        limit: Maximum number of rows to return
        
    Returns:
        Preview of the table data
    """
    preview_df, error = ctx.deps.db_service.get_table_preview(table_name, limit)
    if error:
        return {"error": error}
    
    # Convert the DataFrame to a dictionary for the response
    return {
        "table_name": table_name,
        "preview": preview_df.to_dict(orient="records"),
        "columns": list(preview_df.columns)
    }

@data_chat_agent.tool
def get_column_stats(ctx: RunContext[DBContext], table_name: str, column_name: str) -> Dict[str, Any]:
    """Get basic statistics for a column in a table.
    
    Args:
        table_name: Name of the table
        column_name: Name of the column
        
    Returns:
        Statistics for the column
    """
    if table_name not in ctx.deps.db_service.tables:
        return {"error": f"Table '{table_name}' not found"}
    
    schema = ctx.deps.db_service.tables[table_name]
    if column_name not in schema["columns"]:
        return {"error": f"Column '{column_name}' not found in table '{table_name}'"}
    
    # Get column data type to determine stats to calculate
    col_type = schema["column_types"][column_name]
    
    # Different statistics for numeric vs non-numeric columns
    if "int" in col_type or "float" in col_type:
        # For numeric columns
        query = f"""
        SELECT 
            COUNT({column_name}) as count,
            MIN({column_name}) as min,
            MAX({column_name}) as max,
            AVG({column_name}::FLOAT) as avg,
            STDDEV({column_name}::FLOAT) as stddev
        FROM {table_name}
        """
    else:
        # For non-numeric columns
        query = f"""
        SELECT 
            COUNT({column_name}) as count,
            COUNT(DISTINCT {column_name}) as unique_count
        FROM {table_name}
        """
    
    result_df, error = ctx.deps.db_service.execute_query(query)
    if error:
        return {"error": error}
    
    return result_df.to_dict(orient="records")[0]


# Helpers to process queries and results

def process_query(
    query: str, 
    db_service: DuckDBService, 
    current_table: str
) -> Dict[str, Any]:
    """Process a natural language query and generate SQL.
    
    Args:
        query: Natural language query from the user
        db_service: DuckDB service instance
        current_table: Name of the current active table
        
    Returns:
        Dictionary with generated SQL, explanation, and other metadata
    """
    # Create context for the agent
    context = DBContext(db_service=db_service, current_table=current_table)
    
    # Run the agent to process the query
    result = data_chat_agent.run_sync(
        query, 
        deps=context,
        model_settings={"temperature": agent_settings["temperature"]}
    )
    
    # Return the SQL query and metadata
    return {
        "sql": result.data.sql,
        "explanation": result.data.explanation,
        "visualization_type": result.data.visualization_type,
        "transform_explanation": result.data.transform_explanation
    }

def execute_query_and_analyze(
    sql: str,
    db_service: DuckDBService
) -> Dict[str, Any]:
    """Execute a SQL query and provide analysis of the results.
    
    Args:
        sql: SQL query to execute
        db_service: DuckDB service instance
        
    Returns:
        Dictionary with query results and analysis
    """
    # Execute the query
    result_df, error = db_service.execute_query(sql)
    
    if error:
        return {"success": False, "error": error}
    
    # Basic analysis of the result
    analysis = {
        "row_count": len(result_df),
        "column_count": len(result_df.columns),
        "columns": list(result_df.columns)
    }
    
    return {
        "success": True,
        "result_data": result_df,
        "analysis": analysis
    } 