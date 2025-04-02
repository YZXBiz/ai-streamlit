"""Internal preprocessing assets for the clustering pipeline."""

from typing import Dict

import dagster as dg
import polars as pl

from clustering.core import schemas
from clustering.jobs.utils import validate_dataframe
from clustering.utils.data_processing import clean_ns, create_cat_dict, distribute_sales_evenly, merge_sales_ns


@dg.asset(
    io_manager_key="io_manager",
    key_prefix="raw",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def internal_sales_data(
    context: dg.AssetExecutionContext,
    input_sales_reader=dg.ResourceParam(dg.InitResourceContext),
) -> pl.DataFrame:
    """Load raw internal sales data.

    Args:
        context: Asset execution context
        input_sales_reader: Reader for sales data

    Returns:
        DataFrame containing raw sales data
    """
    context.log.info("Reading sales data")
    sales_data_raw = input_sales_reader.read()

    # Convert to Polars if needed
    if not isinstance(sales_data_raw, pl.DataFrame):
        sales_data = pl.from_pandas(sales_data_raw)
    else:
        sales_data = sales_data_raw

    # Validate the sales data
    sales_data_validated = validate_dataframe(sales_data, schemas.InputsSalesSchema, context.log)

    return sales_data_validated


@dg.asset(
    io_manager_key="io_manager",
    key_prefix="raw",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def internal_need_state_data(
    context: dg.AssetExecutionContext,
    input_need_state_reader=dg.ResourceParam(dg.InitResourceContext),
) -> pl.DataFrame:
    """Load raw internal need state data.

    Args:
        context: Asset execution context
        input_need_state_reader: Reader for need state data

    Returns:
        DataFrame containing cleaned need state data
    """
    context.log.info("Reading need state data")
    ns_data_raw = input_need_state_reader.read()

    # Convert to Polars if needed
    if not isinstance(ns_data_raw, pl.DataFrame):
        ns_data = pl.from_pandas(ns_data_raw)
    else:
        ns_data = ns_data_raw

    # Validate the need state data
    ns_data_validated = validate_dataframe(ns_data, schemas.InputsNSSchema, context.log)

    # Clean need state data
    context.log.info("Cleaning need state data")
    ns_data_cleaned = clean_ns(ns_data_validated)

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
    # Merge data
    context.log.info("Merging sales and need state data")
    merged_data = merge_sales_ns(df_sales=internal_sales_data, df_ns=internal_need_state_data)

    merged_validated = validate_dataframe(merged_data, schemas.InputsMergedSchema, context.log)

    # Redistribute sales
    context.log.info("Redistributing sales based on need states")
    distributed_sales = distribute_sales_evenly(merged_validated)

    distributed_validated = validate_dataframe(distributed_sales, schemas.InputsMergedSchema, context.log)

    return distributed_validated


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
    # Create category dictionary
    context.log.info("Creating category dictionary")
    category_dict = create_cat_dict(merged_internal_data)

    return category_dict


# Define outputs to match the original job outputs
@dg.asset(
    io_manager_key="io_manager",
    deps=["merged_internal_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def preprocessed_internal_sales(
    context: dg.AssetExecutionContext,
    merged_internal_data: pl.DataFrame,
    output_sales_writer=dg.ResourceParam(dg.InitResourceContext),
) -> None:
    """Save preprocessed internal sales data.

    Args:
        context: Asset execution context
        merged_internal_data: Merged and distributed data
        output_sales_writer: Writer for output sales data
    """
    context.log.info("Saving preprocessed sales data")

    # Check if the writer expects Pandas DataFrame
    if hasattr(output_sales_writer, "requires_pandas") and output_sales_writer.requires_pandas:
        output_sales_writer.write(data=merged_internal_data.to_pandas())
    else:
        output_sales_writer.write(data=merged_internal_data)

    return None


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_category_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def preprocessed_internal_sales_percent(
    context: dg.AssetExecutionContext,
    internal_category_data: Dict[str, pl.DataFrame],
    output_sales_percent_writer=dg.ResourceParam(dg.InitResourceContext),
) -> None:
    """Save preprocessed internal sales percentage data.

    Args:
        context: Asset execution context
        internal_category_data: Dictionary of category-specific dataframes
        output_sales_percent_writer: Writer for output sales percent data
    """
    context.log.info("Saving preprocessed sales percent data")

    # Check if the writer expects Pandas DataFrame
    if hasattr(output_sales_percent_writer, "requires_pandas") and output_sales_percent_writer.requires_pandas:
        output_sales_percent_writer.write(data={k: df.to_pandas() for k, df in internal_category_data.items()})
    else:
        output_sales_percent_writer.write(data=internal_category_data)

    return None
