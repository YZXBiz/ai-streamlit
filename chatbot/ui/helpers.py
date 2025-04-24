from typing import Any


def determine_complexity(tables: list[str]) -> str:
    """Automatically determine query complexity based on number of tables.

    Args:
        tables: List of table names available in the database

    Returns:
        str: "advanced" if 2 or more tables exist, "simple" otherwise
    """
    return "advanced" if len(tables) >= 2 else "simple"


def enhance_query_with_context(query: str, tables: list[str]) -> str:
    """Enhance a natural language query with explicit table context.

    Args:
        query: Original user query
        tables: List of available table names

    Returns:
        str: Enhanced query with table context
    """
    table_list = ", ".join(tables)
    table_context = f"Available tables: {table_list}. Only use these tables in your query."

    # Add a note about not using non-existent tables like conversation_history
    note = "Important: Do not reference tables that are not in the above list. Tables like 'conversation_history' do not exist."

    enhanced_query = f"{table_context}\n{note}\n\nQuestion: {query}"
    return enhanced_query


def table_exists(svc: Any, table_name: str) -> bool:
    """Check if a table already exists in the database.

    Args:
        svc: DuckDB service instance
        table_name: Name of the table to check

    Returns:
        bool: True if the table exists, False otherwise
    """
    try:
        # Try to execute a simple query against the table
        svc.execute_query(f"SELECT 1 FROM {table_name} LIMIT 0")
        return True
    except Exception:
        return False
