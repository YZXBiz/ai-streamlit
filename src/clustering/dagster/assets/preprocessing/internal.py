"""Internal preprocessing assets for the clustering pipeline."""

from typing import Any

import dagster as dg
import polars as pl

from clustering.core.sql_engine import DuckDB
from clustering.core.sql_templates import (
    clean_need_state,
    distribute_sales,
    get_categories,
    get_category_data,
    merge_sales_with_need_state,
)


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"input_sales_reader", "config"},
)
def internal_sales_data(
    context: dg.AssetExecutionContext,
) -> pl.DataFrame:
    """Load raw internal sales data.

    Args:
        context: Asset execution context

    Returns:
        DataFrame containing raw sales data
    """
    context.log.info("Reading sales data")

    sales_data_raw = context.resources.input_sales_reader.read()

    # Convert to Polars if needed
    if not isinstance(sales_data_raw, pl.DataFrame):
        sales_data = pl.from_pandas(sales_data_raw)
    else:
        sales_data = sales_data_raw

    # Rename columns to match expected schema
    column_mapping = {
        "product_id": "SKU_NBR",
        "store_id": "STORE_NBR",
        "category_id": "CAT_DSC",
        "sales_amount": "TOTAL_SALES",
    }

    # Only rename columns that exist in the dataframe
    existing_columns = [col for col in column_mapping.keys() if col in sales_data.columns]
    rename_mapping = {col: column_mapping[col] for col in existing_columns}

    sales_data_renamed = sales_data.rename(rename_mapping)

    return sales_data_renamed


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"input_need_state_reader", "config"},
)
def internal_need_state_data(
    context: dg.AssetExecutionContext,
) -> pl.DataFrame:
    """Load raw internal need state data.

    Args:
        context: Asset execution context

    Returns:
        DataFrame containing cleaned need state data
    """
    context.log.info("Reading need state data")

    ns_data_raw = context.resources.input_need_state_reader.read()

    # Convert to Polars if needed
    if not isinstance(ns_data_raw, pl.DataFrame):
        ns_data = pl.from_pandas(ns_data_raw)
    else:
        ns_data = ns_data_raw

    context.log.info("Cleaning need state data")

    # Execute cleaning using DuckDB SQL
    db = DuckDB()
    try:
        clean_sql = clean_need_state(ns_data)
        result = db.query(clean_sql)
    finally:
        db.close()

    return result


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_sales_data", "internal_need_state_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def merged_internal_data(
    context: dg.AssetExecutionContext,
    internal_sales_data: pl.DataFrame,
    internal_need_state_data: pl.DataFrame,
) -> pl.DataFrame:
    """Merge internal sales and need state data.

    Args:
        context: Asset execution context
        internal_sales_data: Sales data
        internal_need_state_data: Need state data

    Returns:
        DataFrame containing merged data with sales distributed evenly
    """
    db = DuckDB()
    try:
        context.log.info("Merging sales and need state data")

        # Validate required columns before creating SQL
        if "PRODUCT_ID" not in internal_need_state_data.columns:
            raise ValueError("Required column 'PRODUCT_ID' missing from need state data")
        if "SKU_NBR" not in internal_sales_data.columns:
            raise ValueError("Required column 'SKU_NBR' missing from sales data")

        # Create and execute merge operation
        merge_sql = merge_sales_with_need_state(
            sales_df=internal_sales_data, need_state_df=internal_need_state_data
        )
        merged_data = db.query(merge_sql)

        context.log.info("Redistributing sales based on need states")

        # Validate required columns for distribution
        required_cols = ["SKU_NBR", "STORE_NBR", "NEED_STATE", "TOTAL_SALES"]
        if not all(col in merged_data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in merged_data.columns]
            raise ValueError(f"Required columns missing: {missing_cols}")

        # Create and execute distribution operation
        distribute_sql = distribute_sales(merged_data)
        distributed_sales = db.query(distribute_sql)

        return distributed_sales
    finally:
        db.close()


@dg.asset(
    io_manager_key="io_manager",
    deps=["merged_internal_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def internal_category_data(
    context: dg.AssetExecutionContext,
    merged_internal_data: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Create category dictionary from merged data.

    Args:
        context: Asset execution context
        merged_internal_data: Merged and distributed data

    Returns:
        Dictionary of category-specific dataframes
    """
    context.log.info("Creating category dictionary")

    # Validate required column
    if "CAT_DSC" not in merged_internal_data.columns:
        raise ValueError("Required column 'CAT_DSC' missing")

    db = DuckDB()
    try:
        # Get unique categories
        categories_sql = get_categories(merged_internal_data)
        categories_result: Any = db.query(categories_sql, output_format="raw")
        categories = [cat[0] for cat in categories_result.fetchall()]

        # Create dictionary by filtering for each category
        cat_dict: dict[str, pl.DataFrame] = {}
        for cat in categories:
            category_sql = get_category_data(merged_internal_data, cat)
            cat_df = db.query(category_sql)
            cat_dict[cat] = cat_df

        return cat_dict
    finally:
        db.close()


@dg.asset(
    io_manager_key="io_manager",
    deps=["merged_internal_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"output_sales_writer"},
)
def preprocessed_internal_sales(
    context: dg.AssetExecutionContext,
    merged_internal_data: pl.DataFrame,
) -> None:
    """Save preprocessed internal sales data.

    Args:
        context: Asset execution context
        merged_internal_data: Merged and distributed data
    """
    context.log.info("Saving preprocessed sales data")

    output_writer = context.resources.output_sales_writer

    # Check if the writer expects Pandas DataFrame
    if hasattr(output_writer, "requires_pandas") and output_writer.requires_pandas:
        output_writer.write(data=merged_internal_data.to_pandas())
    else:
        output_writer.write(data=merged_internal_data)


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_category_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"output_sales_percent_writer"},
)
def preprocessed_internal_sales_percent(
    context: dg.AssetExecutionContext,
    internal_category_data: dict[str, pl.DataFrame],
) -> None:
    """Save preprocessed internal sales percentage data.

    Args:
        context: Asset execution context
        internal_category_data: Dictionary of category-specific dataframes
    """
    context.log.info("Saving preprocessed sales percent data")

    output_writer = context.resources.output_sales_percent_writer

    # Check if the writer expects Pandas DataFrame
    if hasattr(output_writer, "requires_pandas") and output_writer.requires_pandas:
        output_writer.write(data={k: df.to_pandas() for k, df in internal_category_data.items()})
    else:
        output_writer.write(data=internal_category_data)
