"""SQL template functions for the clustering pipeline.

This module contains functions that create SQL objects for common transformations.
Each function returns a SQL object with the query and any bindings needed.
"""

import polars as pl

from clustering.core.sql_engine import SQL
from clustering.core.validation import assert_columns


def clean_need_state(need_state_df: pl.DataFrame) -> SQL:
    """Create SQL to clean need state data.

    Transforms and standardizes the raw need state data into a consistent format
    for merging with sales data.

    Args:
        need_state_df: Need state dataframe with raw data from the CSV file

    Returns:
        SQL object for cleaning need state data

    Raises:
        ValueError: If required columns are missing from the need_state_df
    """
    # Validate that required columns exist
    required_columns = ["category_id", "need_state_id", "need_state_name", "need_state_description"]
    assert_columns(need_state_df, required_columns, context="need state data")

    # Using the actual column names from the CSV file
    return SQL(
        """
        SELECT
            "category_id" AS "CATEGORY_ID",
            "need_state_id" AS "NEED_STATE_ID",
            "need_state_name" AS "NEED_STATE_NAME",
            "need_state_description" AS "NEED_STATE_DESCRIPTION",
            CAST("category_id" AS INTEGER) AS "PRODUCT_ID", -- Using category_id as a temporary substitute for PRODUCT_ID for join
            "need_state_name" AS "NEED_STATE", -- This matches the expected output column from schema
            "need_state_description" AS "CATEGORY", -- Using description as a category stand-in
            'Unknown' AS "CDT", -- Placeholder
            'Unknown' AS "ATTRIBUTE_1",
            'Unknown' AS "ATTRIBUTE_2",  
            'Unknown' AS "ATTRIBUTE_3",
            'Unknown' AS "ATTRIBUTE_4",
            'Unknown' AS "ATTRIBUTE_5",
            'Unknown' AS "ATTRIBUTE_6",
            "need_state_name" AS "PLANOGRAM_DSC", -- Using need_state_name as a placeholder
            "need_state_id" AS "PLANOGRAM_NBR", -- Using need_state_id as a placeholder
            CAST(0 AS BOOLEAN) AS "NEW_ITEM", -- Default value
            CAST(0 AS BOOLEAN) AS "TO_BE_DROPPED" -- Default value
        FROM $need_state
        """,
        bindings={"need_state": need_state_df},
    )


def merge_sales_with_need_state(sales_df: pl.DataFrame, need_state_df: pl.DataFrame) -> SQL:
    """Create SQL to merge sales and need state data.

    Joins the sales data with need state data on the appropriate key columns,
    handling type conversions as needed.

    Args:
        sales_df: Sales dataframe with product and store data
        need_state_df: Cleaned need state dataframe with product categories

    Returns:
        SQL object for merging sales and need state data

    Raises:
        ValueError: If required columns are missing from either DataFrame
    """
    # Validate that required columns exist in sales
    sales_required = ["SKU_NBR", "STORE_NBR", "CAT_DSC", "TOTAL_SALES"]
    assert_columns(sales_df, sales_required, context="sales data")

    # Validate that required columns exist in need state
    need_state_required = ["CATEGORY_ID", "NEED_STATE", "NEED_STATE_DESCRIPTION"]
    assert_columns(need_state_df, need_state_required, context="need state data")

    return SQL(
        """
        SELECT
            s."SKU_NBR",
            s."STORE_NBR",
            s."date" AS "DATE",
            s."CAT_DSC",
            s."need_state_id",
            s."price",
            s."weight",
            s."is_promotional",
            s."is_seasonal",
            s."store_size",
            s."region",
            s."sales_units",
            s."TOTAL_SALES",
            n."NEED_STATE",
            n."NEED_STATE_DESCRIPTION"
        FROM
            $sales s
        JOIN
            $need_state n
        ON
            s."CAT_DSC" = CAST(n."CATEGORY_ID" AS VARCHAR)
        """,
        bindings={"sales": sales_df, "need_state": need_state_df},
    )


def distribute_sales(merged_df: pl.DataFrame) -> SQL:
    """Create SQL to distribute sales based on need states.

    Aggregates sales data by store, SKU, and need state to
    create a summarized view of sales distribution across need states.

    Args:
        merged_df: Merged dataframe containing both sales and need state data

    Returns:
        SQL object for distributing sales

    Raises:
        ValueError: If required columns are missing from the merged_df
    """
    # Validate that required columns exist
    required_columns = [
        "STORE_NBR",
        "SKU_NBR",
        "NEED_STATE",
        "NEED_STATE_DESCRIPTION",
        "TOTAL_SALES",
    ]
    assert_columns(merged_df, required_columns, context="merged data")

    return SQL(
        """
        SELECT
            "STORE_NBR",
            "SKU_NBR",
            "NEED_STATE",
            "NEED_STATE_DESCRIPTION" AS "CAT_DSC",
            SUM("TOTAL_SALES") AS "TOTAL_SALES"
        FROM
            $merged_data
        GROUP BY
            "STORE_NBR", "SKU_NBR", "NEED_STATE", "NEED_STATE_DESCRIPTION"
        """,
        bindings={"merged_data": merged_df},
    )


def get_categories(data_df: pl.DataFrame) -> SQL:
    """Create SQL to get distinct categories.

    Extracts the unique category values from the data for further processing.

    Args:
        data_df: Data dataframe with a CAT_DSC column

    Returns:
        SQL object for getting categories

    Raises:
        ValueError: If the required CAT_DSC column is missing
    """
    # Validate that required column exists
    assert_columns(data_df, ["CAT_DSC"], context="data for categories")

    return SQL(
        """
        SELECT DISTINCT "CAT_DSC" 
        FROM $data
        """,
        bindings={"data": data_df},
    )


def get_category_data(data_df: pl.DataFrame, category_value: str) -> SQL:
    """Create SQL to get data for a specific category.

    Filters the data to only include rows for a specific category value.

    Args:
        data_df: Data dataframe with a CAT_DSC column
        category_value: Category value to filter by

    Returns:
        SQL object for getting category data

    Raises:
        ValueError: If the required CAT_DSC column is missing
    """
    # Validate that required column exists
    assert_columns(data_df, ["CAT_DSC"], context="data for category filtering")

    return SQL(
        """
        SELECT *
        FROM $data
        WHERE "CAT_DSC" = $category_value
        """,
        bindings={"data": data_df, "category_value": category_value},
    )
