"""
Result interpreter for data query results.

This module provides functions for interpreting and explaining the results of
data queries in natural language.
"""

import re
from typing import Any

import pandas as pd

from assortment_chatbot.utils.log_config import get_logger

logger = get_logger(__name__)


def interpret_results(
    original_query: str, sql_query: str, result_df: pd.DataFrame, is_fallback: bool = False
) -> str:
    """
    Interpret the results of a query and provide a natural language explanation.

    Args:
        original_query: The original natural language query
        sql_query: The SQL query that was executed
        result_df: The DataFrame containing the query results
        is_fallback: Whether this is a fallback interpretation (from a simplified query)

    Returns:
        A natural language explanation of the results
    """
    try:
        # Get basic statistics from the DataFrame
        row_count = len(result_df)

        if row_count == 0:
            return "The query returned no results."

        # Start with a different introduction for fallback queries
        if is_fallback:
            explanation = "I couldn't answer your specific question, but here's some general information about your data:\n\n"
        else:
            explanation = f"I found {row_count} rows matching your query."

        # Add information about columns
        explanation += f"\n\nThe data includes {len(result_df.columns)} columns: "
        explanation += ", ".join(f"`{col}`" for col in result_df.columns)

        # Add sample data if there's at least one row
        if row_count > 0:
            explanation += "\n\nHere's a summary of the results:"

            # For numeric columns, include min, max, mean
            numeric_cols = result_df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                explanation += "\n\n**Numeric summary:**"
                for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                    explanation += f"\n- `{col}`: min={result_df[col].min():.2f}, max={result_df[col].max():.2f}, avg={result_df[col].mean():.2f}"

            # For categorical columns, include top values and counts
            categorical_cols = result_df.select_dtypes(exclude=["number"]).columns
            if len(categorical_cols) > 0:
                explanation += "\n\n**Categorical summary:**"
                for col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    if row_count > 0 and len(result_df[col].value_counts()) > 0:
                        top_value = result_df[col].value_counts().index[0]
                        top_count = result_df[col].value_counts().iloc[0]
                        explanation += f"\n- `{col}`: most common value is '{top_value}' ({top_count} occurrences)"

            # Try to detect if this was an aggregation query
            if "COUNT(" in sql_query.upper() and row_count == 1 and len(result_df.columns) == 1:
                count_val = result_df.iloc[0, 0]
                explanation = f"There are {count_val} records in total."

            elif "AVG(" in sql_query.upper() and row_count == 1:
                # Extract the column name being averaged
                match = re.search(r"AVG\((\w+)\)", sql_query, re.IGNORECASE)
                if match:
                    col_name = match.group(1)
                    avg_val = result_df.iloc[0, 0]
                    explanation = f"The average {col_name} is {avg_val:.2f}."

            # Try to extract the main subject from the original query
            if "what is" in original_query.lower() or "how many" in original_query.lower():
                # Very simplistic approach - a real implementation would use NLP
                pass

            # Add overall data insights based on the query structure
            if not is_fallback:
                explanation += "\n\n**Key insights:**"

                # For aggregations like COUNT, SUM, AVG, etc.
                if any(
                    agg in sql_query.upper() for agg in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("]
                ):
                    # This is a placeholder - real implementation would analyze the specific aggregation
                    explanation += "\n- This aggregation query provides a statistical overview."

                # For filtered queries
                elif "WHERE" in sql_query.upper():
                    explanation += "\n- This filtered dataset represents a subset of all records."

                # For grouped queries
                elif "GROUP BY" in sql_query.upper():
                    explanation += (
                        "\n- The data is organized by groups, useful for comparing categories."
                    )

        return explanation

    except Exception:
        logger.error("Error interpreting results", exc_info=True)
        return f"I found {len(result_df)} results from your query, but couldn't generate a detailed analysis."


def verify_interpretation(
    interpretation: str, result_df: pd.DataFrame, sql_query: str
) -> dict[str, Any]:
    """
    Verify that an interpretation correctly describes the results and correct if needed.

    Args:
        interpretation: The generated interpretation
        result_df: The DataFrame containing the query results
        sql_query: The SQL query that was executed

    Returns:
        Dict with verification status and possibly corrected interpretation
    """
    verification = {"verified": True, "corrected_interpretation": interpretation, "issues": []}

    try:
        # Check for basic numerical accuracy
        row_count = len(result_df)

        # Check if row count is mentioned correctly
        row_count_match = re.search(r"I found (\d+) rows", interpretation)
        if row_count_match:
            mentioned_count = int(row_count_match.group(1))
            if mentioned_count != row_count:
                verification["verified"] = False
                verification["issues"].append(
                    f"Incorrect row count: {mentioned_count} vs actual {row_count}"
                )

                # Fix the row count
                verification["corrected_interpretation"] = interpretation.replace(
                    f"I found {mentioned_count} rows", f"I found {row_count} rows"
                )

        # Check column count accuracy
        col_count_match = re.search(r"data includes (\d+) columns", interpretation)
        if col_count_match:
            mentioned_col_count = int(col_count_match.group(1))
            actual_col_count = len(result_df.columns)
            if mentioned_col_count != actual_col_count:
                verification["verified"] = False
                verification["issues"].append(
                    f"Incorrect column count: {mentioned_col_count} vs actual {actual_col_count}"
                )

                # Fix the column count
                if "corrected_interpretation" in verification:
                    verification["corrected_interpretation"] = verification[
                        "corrected_interpretation"
                    ].replace(
                        f"data includes {mentioned_col_count} columns",
                        f"data includes {actual_col_count} columns",
                    )
                else:
                    verification["corrected_interpretation"] = interpretation.replace(
                        f"data includes {mentioned_col_count} columns",
                        f"data includes {actual_col_count} columns",
                    )

        # Check numeric summaries for accuracy
        # This is a simplified check - a real implementation would be more thorough
        numeric_cols = result_df.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            if col in interpretation:
                # Check if min value is correctly stated
                min_match = re.search(f"`{col}`.*?min=([0-9.]+)", interpretation)
                if min_match:
                    mentioned_min = float(min_match.group(1))
                    actual_min = round(result_df[col].min(), 2)

                    if (
                        abs(mentioned_min - actual_min) > 0.01
                    ):  # Allow small floating point differences
                        verification["verified"] = False
                        verification["issues"].append(
                            f"Incorrect min for {col}: {mentioned_min} vs actual {actual_min}"
                        )

                        # Fix the min value
                        if "corrected_interpretation" in verification:
                            verification["corrected_interpretation"] = re.sub(
                                f"`{col}`.*?min=([0-9.]+)",
                                f"`{col}`: min={actual_min:.2f}",
                                verification["corrected_interpretation"],
                            )
                        else:
                            verification["corrected_interpretation"] = re.sub(
                                f"`{col}`.*?min=([0-9.]+)",
                                f"`{col}`: min={actual_min:.2f}",
                                interpretation,
                            )

                # Similar checks could be implemented for max and avg

        # Check for major statement inaccuracies or contradictions
        # This would require more sophisticated NLP in a real implementation

        return verification

    except Exception as e:
        logger.error("Error verifying interpretation", exc_info=True)
        # Return the original interpretation if verification fails
        return {
            "verified": False,
            "corrected_interpretation": interpretation,
            "issues": [f"Verification error: {str(e)}"],
        }
