"""Result interpretation agent for SQL query results.

This module provides AI capabilities to analyze and interpret the results
of SQL queries in natural language, making data insights more accessible.
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from dashboard.settings import get_settings

# Get settings
settings = get_settings()

class ResultInterpretation(BaseModel):
    """Interpretation of SQL query results."""
    
    summary: str = Field(description="High-level summary of the query results")
    insights: List[str] = Field(description="Key insights from the data")
    recommendations: Optional[List[str]] = Field(
        description="Recommended next steps or further analyses",
        default=None
    )


@dataclass
class ResultContext:
    """Context for result interpretation."""
    original_query: str
    sql_query: str
    result_df: pd.DataFrame
    column_descriptions: Optional[Dict[str, str]] = None


# Create the PydanticAI agent for interpreting results
result_interpreter_agent = Agent(
    settings.agent.interpreter_model,  # Get model name from settings
    deps_type=ResultContext,
    result_type=ResultInterpretation,
    system_prompt="""
    You are an expert data analyst who explains SQL query results in clear, 
    natural language. Your role is to:
    
    1. Provide a clear, concise summary of what the results show
    2. Identify key insights and patterns in the data
    3. Suggest potential next steps or follow-up analyses
    
    Focus on making complex data understandable to non-technical users.
    Highlight important trends, anomalies, or interesting findings.
    """,
    model_settings={"temperature": settings.agent.temperature}
)


@result_interpreter_agent.tool
def get_result_stats(ctx: RunContext[ResultContext]) -> Dict[str, Any]:
    """Get basic statistics about the query results.
    
    Returns statistical information about the dataframe.
    """
    df = ctx.deps.result_df
    
    # Basic stats about the result
    stats = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns)
    }
    
    # Add numeric column statistics if available
    numeric_cols = df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        stats["numeric_stats"] = {}
        for col in numeric_cols:
            stats["numeric_stats"][col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
            }
    
    return stats


@result_interpreter_agent.tool
def get_result_sample(ctx: RunContext[ResultContext], limit: int = None) -> Dict[str, Any]:
    """Get a sample of rows from the result.
    
    Args:
        limit: Maximum number of rows to return. If None, uses settings default.
        
    Returns:
        Sample rows from the result
    """
    df = ctx.deps.result_df
    
    # Use settings for default limit if not specified
    if limit is None:
        limit = settings.duckdb.max_rows_preview
    
    # Get a sample of rows
    sample = df.head(limit).to_dict(orient="records")
    
    return {
        "sample": sample,
        "columns": list(df.columns)
    }


def interpret_results(
    original_query: str,
    sql_query: str,
    result_df: pd.DataFrame,
    column_descriptions: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Interpret SQL query results and provide natural language insights.
    
    Args:
        original_query: Original natural language query from the user
        sql_query: SQL query that was executed
        result_df: DataFrame containing the query results
        column_descriptions: Optional descriptions of column meanings
        
    Returns:
        Dictionary with interpretation of the results
    """
    # Create context for the agent
    context = ResultContext(
        original_query=original_query,
        sql_query=sql_query,
        result_df=result_df,
        column_descriptions=column_descriptions
    )
    
    # Run the agent to interpret the results
    result = result_interpreter_agent.run_sync(
        f"Interpret the results of the query: {original_query}",
        deps=context,
        model_settings={"temperature": settings.agent.temperature}
    )
    
    # Return the interpretation
    return {
        "summary": result.data.summary,
        "insights": result.data.insights,
        "recommendations": result.data.recommendations or []
    } 