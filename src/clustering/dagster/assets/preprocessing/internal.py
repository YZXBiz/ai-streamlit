"""Internal preprocessing assets for the clustering pipeline."""

import dagster as dg
import polars as pl
from dagster_pandera import pandera_schema_to_dagster_type

from clustering.core.schemas import (
    DistributedDataSchema,
    MergedDataSchema,
    NSMappingSchema,
    SalesSchema,
)


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"internal_ns_sales"},
    dagster_type=pandera_schema_to_dagster_type(SalesSchema),
)
def internal_raw_sales_data(context: dg.AssetExecutionContext) -> pl.DataFrame:
    """Load raw internal sales data.

    Args:
        context: Asset execution context

    Returns:
        DataFrame containing raw sales data
    """
    context.log.info("Reading need state sales data")

    # Convert back to polars
    return context.resources.internal_ns_sales.read()


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"internal_ns_map"},
    dagster_type=pandera_schema_to_dagster_type(NSMappingSchema),
)
def internal_product_category_mapping(
    context: dg.AssetExecutionContext,
) -> pl.DataFrame:
    """Load raw internal need state data.

    Args:
        context: Asset execution context

    Returns:
        DataFrame containing cleaned product category mapping data
    """
    return (
        context.resources.internal_ns_map.read()
        .filter(pl.col("PRODUCT_ID").is_not_null())
        .with_columns(pl.col("NEED_STATE").str.to_uppercase())
        .unique()
    )


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_raw_sales_data", "internal_product_category_mapping"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    dagster_type=pandera_schema_to_dagster_type(MergedDataSchema),
)
def internal_sales_with_categories(
    context: dg.AssetExecutionContext,
    internal_raw_sales_data: pl.DataFrame,
    internal_product_category_mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Merge sales data with product category mapping.

    Args:
        context: Asset execution context
        internal_raw_sales_data: Sales data
        internal_product_category_mapping: Product category mapping data

    Returns:
        DataFrame containing merged data with categories
    """
    context.log.info("Merging sales and category data")

    return internal_raw_sales_data.join(
        internal_product_category_mapping,
        left_on="SKU_NBR",
        right_on="PRODUCT_ID",
        how="inner",
    ).select(
        "SKU_NBR",
        "STORE_NBR",
        "CAT_DSC",
        "NEED_STATE",
        "TOTAL_SALES",
    )


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_sales_with_categories"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    dagster_type=pandera_schema_to_dagster_type(DistributedDataSchema),
)
def internal_normalized_sales_data(
    context: dg.AssetExecutionContext,
    internal_sales_with_categories: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize sales data by distributing sales evenly across need states.

    Args:
        context: Asset execution context
        internal_sales_with_categories: Sales data with categories

    Returns:
        DataFrame containing normalized sales data
    """
    context.log.info("Distributing sales evenly across need states")

    return (
        internal_sales_with_categories.pipe(
            lambda df: df.with_columns(
                pl.count()
                .over([c for c in df.columns if c != "NEED_STATE" and c != "TOTAL_SALES"])
                .alias("group_count")
            )
        )
        .with_columns(
            (pl.col("TOTAL_SALES") / pl.col("group_count")).alias("TOTAL_SALES"),
        )
        .drop("group_count")
    )


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_normalized_sales_data"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def internal_sales_by_category(
    context: dg.AssetExecutionContext,
    internal_normalized_sales_data: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Create category dictionary from normalized sales data with percentage of sales by need state.

    This asset:
    1. Groups data by category
    2. For each category, calculates store-need state sales and store total sales
    3. Computes percentage of sales by need state for each store
    4. Pivots the data to have need states as columns

    Args:
        context: Asset execution context
        internal_normalized_sales_data: Normalized sales data with categories

    Returns:
        Dictionary of category-specific dataframes with need state percentage metrics
    """
    context.log.info("Creating category dictionary with need state percentage metrics")

    # Get unique categories
    categories = (
        internal_normalized_sales_data.select(pl.col("CAT_DSC").unique()).to_series().to_list()
    )

    # Create result dictionary
    result = {}

    # Process each category
    for cat in categories:
        # Filter data for this category
        cat_data = internal_normalized_sales_data.filter(pl.col("CAT_DSC") == cat)

        # Calculate store-need state sales
        store_ns_sales = cat_data.group_by(["STORE_NBR", "NEED_STATE"]).agg(
            pl.sum("TOTAL_SALES").alias("STORE_NS_TOTAL_SALES")
        )

        # Calculate store total sales
        store_sales = cat_data.group_by("STORE_NBR").agg(
            pl.sum("TOTAL_SALES").alias("STORE_TOTAL_SALES")
        )

        # Merge and calculate percentages
        merged = store_ns_sales.join(store_sales, on="STORE_NBR", how="left").with_columns(
            (pl.col("STORE_NS_TOTAL_SALES") / pl.col("STORE_TOTAL_SALES") * 100.0).alias(
                "Pct_of_Sales"
            )
        )

        # Get need states for column renaming
        need_states = merged.select("NEED_STATE").unique().to_series().to_list()

        # Pivot the data
        pivoted = merged.pivot(
            index="STORE_NBR", columns="NEED_STATE", values="Pct_of_Sales"
        ).fill_null(0)

        # Create column rename mapping
        rename_map = {ns: f"% Sales {ns}" for ns in need_states}

        # Rename columns and round values
        final_df = pivoted.rename(rename_map)

        # Round all percentage columns
        round_cols = [f"% Sales {ns}" for ns in need_states]
        final_df = final_df.with_columns([pl.col(col).round(2) for col in round_cols])

        # Add to result dictionary
        result[cat] = final_df

    return result


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_sales_by_category"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"sales_by_category_writer"},
)
def internal_output_sales_table(
    context: dg.AssetExecutionContext,
    internal_sales_by_category: dict[str, pl.DataFrame],
) -> None:
    """Save preprocessed sales data to output.

    Args:
        context: Asset execution context
        internal_sales_by_category: Sales data with categories
    """
    context.log.info("Saving preprocessed sales data")
    context.resources.sales_by_category_writer.write(data=internal_sales_by_category)

    # Collect all unique store numbers across all categories
    all_stores = set()
    for category_df in internal_sales_by_category.values():
        stores = category_df.select("STORE_NBR").unique().to_series().to_list()
        all_stores.update(stores)

    context.add_output_metadata(
        metadata={
            "num_categories": len(internal_sales_by_category),
            "num_stores": len(all_stores),
        }
    )
