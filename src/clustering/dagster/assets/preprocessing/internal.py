"""Internal preprocessing assets for the clustering pipeline."""

from typing import Dict

import dagster as dg
import polars as pl

from clustering.core import schemas
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
    key_prefix="raw",
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

    # Use the dedicated reader resource
    sales_data_raw = context.resources.input_sales_reader.read()

    # Convert to Polars if needed
    if not isinstance(sales_data_raw, pl.DataFrame):
        sales_data = pl.from_pandas(sales_data_raw)
    else:
        sales_data = sales_data_raw

    # Validate the sales data
    sales_data_validated = schemas.InputsSalesSchema.check(sales_data)

    return sales_data_validated


@dg.asset(
    io_manager_key="io_manager",
    key_prefix="raw",
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

    # Use the dedicated reader resource
    ns_data_raw = context.resources.input_need_state_reader.read()

    # Convert to Polars if needed
    if not isinstance(ns_data_raw, pl.DataFrame):
        ns_data = pl.from_pandas(ns_data_raw)
    else:
        ns_data = ns_data_raw

    # Validate the need state data
    ns_data_validated = schemas.InputsNSSchema.check(ns_data)

    # Clean need state data using DuckDB SQL
    context.log.info("Cleaning need state data")

    # First rename columns to uppercase
    column_mapping = {col: col.upper() for col in ns_data_validated.columns}
    ns_data_upper = ns_data_validated.rename(column_mapping)

    # Execute cleaning using the functional SQL approach
    db = DuckDB()
    try:
        # Create a SQL object for the cleaning operation
        clean_sql = clean_need_state(ns_data_upper)
        # Execute the query and convert to a DataFrame
        result = db.query(clean_sql)

        # Extract original columns to avoid duplicates from CASE expressions
        cols = list(ns_data_upper.columns)
        ns_data_cleaned = result.select(cols)
    finally:
        db.close()

    return ns_data_cleaned


@dg.asset(
    io_manager_key="io_manager",
    deps=["raw/internal_sales_data", "raw/internal_need_state_data"],
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
    # Use the functional SQL approach for transforms
    db = DuckDB()
    try:
        # Merge data using a SQL object
        context.log.info("Merging sales and need state data")

        # Validate required columns before creating SQL
        if "PRODUCT_ID" not in internal_need_state_data.columns:
            raise ValueError("Required column 'PRODUCT_ID' missing from need state data")
        if "SKU_NBR" not in internal_sales_data.columns:
            raise ValueError("Required column 'SKU_NBR' missing from sales data")

        # Create a SQL object for the merge operation
        merge_sql = merge_sales_with_need_state(sales_df=internal_sales_data, need_state_df=internal_need_state_data)

        # Execute the query
        merged_data = db.query(merge_sql)
        merged_validated = schemas.InputsMergedSchema.check(merged_data)

        # Redistribute sales using a SQL object
        context.log.info("Redistributing sales based on need states")

        # Validate required columns for distribution
        required_cols = ["SKU_NBR", "STORE_NBR", "NEED_STATE", "TOTAL_SALES"]
        if not all(col in merged_validated.columns for col in required_cols):
            raise ValueError(f"Required columns missing: {required_cols}")

        # Create a SQL object for the distribution operation
        distribute_sql = distribute_sales(merged_validated)

        # Execute the query
        distributed_sales = db.query(distribute_sql)
        distributed_validated = schemas.InputsMergedSchema.check(distributed_sales)

        return distributed_validated
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
) -> Dict[str, pl.DataFrame]:
    """Create category dictionary from merged data.

    Args:
        context: Asset execution context
        merged_internal_data: Merged and distributed data

    Returns:
        Dictionary of category-specific dataframes
    """
    # Create category dictionary using functional SQL approach
    context.log.info("Creating category dictionary")

    # Validate required column
    if "CAT_DSC" not in merged_internal_data.columns:
        raise ValueError("Required column 'CAT_DSC' missing")

    db = DuckDB()
    try:
        # Get unique categories
        categories_sql = get_categories(merged_internal_data)
        categories_result = db.query(categories_sql, output_format="raw")
        categories = [cat[0] for cat in categories_result.fetchall()]

        # Create dictionary by filtering for each category
        cat_dict = {}
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

    # Use the configured writer resource
    output_writer = context.resources.output_sales_writer

    # Check if the writer expects Pandas DataFrame
    if hasattr(output_writer, "requires_pandas") and output_writer.requires_pandas:
        output_writer.write(data=merged_internal_data.to_pandas())
    else:
        output_writer.write(data=merged_internal_data)

    return None


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_category_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"output_sales_percent_writer"},
)
def preprocessed_internal_sales_percent(
    context: dg.AssetExecutionContext,
    internal_category_data: Dict[str, pl.DataFrame],
) -> None:
    """Save preprocessed internal sales percentage data.

    Args:
        context: Asset execution context
        internal_category_data: Dictionary of category-specific dataframes
    """
    context.log.info("Saving preprocessed sales percent data")

    # Use the configured writer resource
    output_writer = context.resources.output_sales_percent_writer

    # Check if the writer expects Pandas DataFrame
    if hasattr(output_writer, "requires_pandas") and output_writer.requires_pandas:
        output_writer.write(data={k: df.to_pandas() for k, df in internal_category_data.items()})
    else:
        output_writer.write(data=internal_category_data)

    return None
