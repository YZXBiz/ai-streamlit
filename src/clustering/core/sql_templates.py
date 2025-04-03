"""SQL template functions for the clustering pipeline.

This module contains functions that create SQL objects for common transformations.
Each function returns a SQL object with the query and any bindings needed.
"""

import polars as pl

from clustering.core.sql_engine import SQL


def clean_need_state(need_state_df: pl.DataFrame) -> SQL:
    """Create SQL to clean need state data.

    Args:
        need_state_df: Need state dataframe

    Returns:
        SQL object for cleaning need state data
    """
    return SQL(
        """
        SELECT
            *,
            CASE 
                WHEN "ATTRIBUTE_1" IS NULL THEN 'UNKNOWN' 
                ELSE "ATTRIBUTE_1" 
            END AS "ATTRIBUTE_1",
            CASE 
                WHEN "ATTRIBUTE_2" IS NULL THEN 'UNKNOWN' 
                ELSE "ATTRIBUTE_2" 
            END AS "ATTRIBUTE_2",
            CASE 
                WHEN "ATTRIBUTE_3" IS NULL THEN 'UNKNOWN' 
                ELSE "ATTRIBUTE_3" 
            END AS "ATTRIBUTE_3",
            CASE 
                WHEN "ATTRIBUTE_4" IS NULL THEN 'UNKNOWN' 
                ELSE "ATTRIBUTE_4" 
            END AS "ATTRIBUTE_4",
            CASE 
                WHEN "ATTRIBUTE_5" IS NULL THEN 'UNKNOWN' 
                ELSE "ATTRIBUTE_5" 
            END AS "ATTRIBUTE_5",
            CASE 
                WHEN "ATTRIBUTE_6" IS NULL THEN 'UNKNOWN' 
                ELSE "ATTRIBUTE_6" 
            END AS "ATTRIBUTE_6",
            CAST("NEW_ITEM" AS BOOLEAN) AS "NEW_ITEM",
            CAST("TO_BE_DROPPED" AS BOOLEAN) AS "TO_BE_DROPPED"
        FROM $need_state
        """,
        bindings={"need_state": need_state_df},
    )


def merge_sales_with_need_state(sales_df: pl.DataFrame, need_state_df: pl.DataFrame) -> SQL:
    """Create SQL to merge sales and need state data.

    Args:
        sales_df: Sales dataframe
        need_state_df: Need state dataframe

    Returns:
        SQL object for merging sales and need state data
    """
    return SQL(
        """
        SELECT
            s.*,
            n."NEED_STATE"
        FROM
            $sales s
        JOIN
            $need_state n
        ON
            s."SKU_NBR" = n."PRODUCT_ID"
        """,
        bindings={"sales": sales_df, "need_state": need_state_df},
    )


def distribute_sales(merged_df: pl.DataFrame) -> SQL:
    """Create SQL to distribute sales based on need states.

    Args:
        merged_df: Merged dataframe

    Returns:
        SQL object for distributing sales
    """
    return SQL(
        """
        SELECT
            "STORE_NBR",
            "SKU_NBR",
            "NEED_STATE",
            "CAT_DSC",
            SUM("TOTAL_SALES") AS "TOTAL_SALES"
        FROM
            $merged_data
        GROUP BY
            "STORE_NBR", "SKU_NBR", "NEED_STATE", "CAT_DSC"
        """,
        bindings={"merged_data": merged_df},
    )


def get_categories(data_df: pl.DataFrame) -> SQL:
    """Create SQL to get distinct categories.

    Args:
        data_df: Data dataframe

    Returns:
        SQL object for getting categories
    """
    return SQL(
        """
        SELECT DISTINCT "CAT_DSC" 
        FROM $data
        """,
        bindings={"data": data_df},
    )


def get_category_data(data_df: pl.DataFrame, category_value: str) -> SQL:
    """Create SQL to get data for a specific category.

    Args:
        data_df: Data dataframe
        category_value: Category value to filter by

    Returns:
        SQL object for getting category data
    """
    return SQL(
        """
        SELECT *
        FROM $data
        WHERE "CAT_DSC" = $category_value
        """,
        bindings={"data": data_df, "category_value": category_value},
    )
