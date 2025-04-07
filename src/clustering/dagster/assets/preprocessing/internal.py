"""Internal preprocessing assets for the clustering pipeline."""

import dagster as dg
import polars as pl


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"input_sales_reader"},
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

    # Define column mapping
    column_mapping = {
        "product_id": "SKU_NBR",
        "store_id": "STORE_NBR",
        "category_id": "CAT_DSC",
        "sales_amount": "TOTAL_SALES",
    }

    # Read data, convert to Polars, and rename columns in a chain
    return (
        context.resources.input_sales_reader.read()
        .pipe(lambda df: pl.from_pandas(df) if not isinstance(df, pl.DataFrame) else df)
        .pipe(
            lambda df: df.rename(
                {col: column_mapping[col] for col in column_mapping.keys() if col in df.columns}
            )
        )
    )


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"input_need_state_reader"},
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
    # Read data and convert to Polars if needed in a chain
    return (
        context.resources.input_need_state_reader.read()
        .pipe(lambda df: pl.from_pandas(df) if not isinstance(df, pl.DataFrame) else df)
        .filter(pl.col("PRODUCT_ID").is_not_null())
        .with_columns(pl.col("NEED_STATE").str.to_uppercase())
        .unique()
    )


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
    # Validate required columns before merging
    if "PRODUCT_ID" not in internal_need_state_data.columns:
        raise ValueError("Required column 'PRODUCT_ID' missing from need state data")
    if "SKU_NBR" not in internal_sales_data.columns:
        raise ValueError("Required column 'SKU_NBR' missing from sales data")

    context.log.info("Merging sales and need state data")
    context.log.info("Redistributing sales based on need states")

    # Merge data
    merged_data = internal_sales_data.join(
        internal_need_state_data, left_on="SKU_NBR", right_on="PRODUCT_ID", how="inner"
    )

    # Validate required columns for distribution
    required_cols = ["SKU_NBR", "STORE_NBR", "NEED_STATE", "TOTAL_SALES"]
    missing_cols = [col for col in required_cols if col not in merged_data.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    # Chain distribution operations
    return (
        merged_data.group_by(["SKU_NBR", "STORE_NBR", "NEED_STATE"])
        .agg(pl.col("TOTAL_SALES").sum().alias("TOTAL_SALES"))
        .with_columns(
            pl.col("TOTAL_SALES")
            / pl.col("TOTAL_SALES").sum().over(["SKU_NBR", "STORE_NBR"]).alias("SALES_PCT")
        )
    )


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

    # Validate required column before proceeding
    if "CAT_DSC" not in merged_internal_data.columns:
        raise ValueError("Required column 'CAT_DSC' missing")

    # Get categories and create dictionary in a chain
    return (
        merged_internal_data.select(pl.col("CAT_DSC").unique())
        .to_series()
        .to_list()
        .pipe(
            lambda categories: {
                cat: merged_internal_data.filter(pl.col("CAT_DSC") == cat) for cat in categories
            }
        )
    )


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

    # Convert data if needed based on writer requirements
    data_to_write = (
        merged_internal_data.to_pandas()
        if hasattr(context.resources.output_sales_writer, "requires_pandas")
        and context.resources.output_sales_writer.requires_pandas
        else merged_internal_data
    )

    # Write the data
    context.resources.output_sales_writer.write(data=data_to_write)


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

    # Convert data if needed based on writer requirements
    data_to_write = (
        {k: df.to_pandas() for k, df in internal_category_data.items()}
        if hasattr(context.resources.output_sales_percent_writer, "requires_pandas")
        and context.resources.output_sales_percent_writer.requires_pandas
        else internal_category_data
    )

    # Write the data
    context.resources.output_sales_percent_writer.write(data=data_to_write)
