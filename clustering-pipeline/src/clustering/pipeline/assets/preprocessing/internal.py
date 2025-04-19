"""Internal preprocessing assets for the clustering pipeline."""

import os
import traceback

import dagster as dg
import pandas as pd
import polars as pl



@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
)
def internal_raw_sales_data(context: dg.AssetExecutionContext) -> pl.DataFrame:
    """Load raw internal sales data.

    Args:
        context: Asset execution context

    Returns:
        DataFrame containing raw sales data
    """
    context.log.info("Reading need state sales data")

    try:
        # Get the file path from resources config
        file_path = "/workspaces/clustering-dagster/data/internal/ns_sales.csv"

        # Check if file exists
        if not os.path.exists(file_path):
            context.log.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        context.log.info(f"Reading from file: {file_path}")

        # Read file directly with polars with careful error handling
        try:
            df = pl.read_csv(file_path)
        except Exception as e:
            context.log.warning(f"Polars read failed: {str(e)}. Trying pandas fallback.")
            # Fallback to pandas
            pandas_df = pd.read_csv(file_path)
            df = pl.from_pandas(pandas_df)

        context.log.info(f"Successfully read data with shape: {df.shape}")

        # Ensure required columns are present
        required_columns = ["SKU_NBR", "STORE_NBR", "CAT_DSC", "TOTAL_SALES"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            context.log.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Sales data is missing required columns: {missing_columns}")

        # Ensure correct types
        df = df.with_columns(
            [
                pl.col("SKU_NBR").cast(pl.Int64),
                pl.col("STORE_NBR").cast(pl.Int64),
                pl.col("CAT_DSC").cast(pl.Utf8),
                pl.col("TOTAL_SALES").cast(pl.Float64),
            ]
        )

        context.log.info(f"Successfully processed sales data: {df.shape}")
        return df

    except Exception as e:
        context.log.error(f"Error processing sales data: {str(e)}")
        context.log.error(f"Exception type: {type(e).__name__}")
        context.log.error(f"Traceback: {traceback.format_exc()}")
        raise


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
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
    try:
        # Get the file path directly
        file_path = "/workspaces/clustering-dagster/data/internal/ns_map.csv"

        # Check if file exists
        if not os.path.exists(file_path):
            context.log.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        context.log.info(f"Reading from file: {file_path}")

        # Read file directly with careful error handling
        try:
            df = pl.read_csv(file_path)
        except Exception as e:
            context.log.warning(f"Polars read failed: {str(e)}. Trying pandas fallback.")
            # Fallback to pandas
            pandas_df = pd.read_csv(file_path)
            df = pl.from_pandas(pandas_df)

        context.log.info(f"Successfully read data with shape: {df.shape}, columns: {df.columns}")

        # Check for required column
        if "PRODUCT_ID" not in df.columns:
            context.log.error("Missing required column: PRODUCT_ID")
            raise ValueError("Product mapping data is missing required column: PRODUCT_ID")

        # Process the data
        context.log.info("Filtering out null PRODUCT_ID rows")
        result_df = df.filter(pl.col("PRODUCT_ID").is_not_null())

        # Add NEED_STATE column if missing
        if "NEED_STATE" not in result_df.columns:
            context.log.warning("NEED_STATE column not found in data, creating default values")
            result_df = result_df.with_columns(pl.lit("DEFAULT").alias("NEED_STATE"))
        else:
            # Convert to uppercase
            result_df = result_df.with_columns(pl.col("NEED_STATE").str.to_uppercase())

        # Ensure PRODUCT_ID is an integer
        result_df = result_df.with_columns(pl.col("PRODUCT_ID").cast(pl.Int64))

        # Remove duplicates
        result_df = result_df.unique()

        context.log.info(f"Successfully processed mapping data: {result_df.shape}")
        return result_df

    except Exception as e:
        context.log.error(f"Error processing product mapping data: {str(e)}")
        context.log.error(f"Exception type: {type(e).__name__}")
        context.log.error(f"Traceback: {traceback.format_exc()}")
        raise


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_raw_sales_data", "internal_product_category_mapping"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
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

    try:
        context.log.info(
            f"Sales data: {internal_raw_sales_data.shape}, Mapping data: {internal_product_category_mapping.shape}"
        )

        # Handle column renames if needed
        if "SKU_NBR" not in internal_raw_sales_data.columns:
            context.log.error("SKU_NBR column not found in sales data")
            raise ValueError("SKU_NBR column not found in sales data")

        # Ensure mapping has correct join column
        if "PRODUCT_ID" not in internal_product_category_mapping.columns:
            context.log.error("PRODUCT_ID column not found in mapping data")
            raise ValueError("PRODUCT_ID column not found in mapping data")

        # Perform join
        merged_df = internal_raw_sales_data.join(
            internal_product_category_mapping,
            left_on="SKU_NBR",
            right_on="PRODUCT_ID",
            how="inner",
        )

        context.log.info(f"Join resulted in {merged_df.shape[0]} rows")

        # Select required columns
        result_columns = ["SKU_NBR", "STORE_NBR", "CAT_DSC", "NEED_STATE", "TOTAL_SALES"]
        missing_columns = [col for col in result_columns if col not in merged_df.columns]

        if missing_columns:
            context.log.error(f"Merged data missing required columns: {missing_columns}")
            raise ValueError(f"Merged data missing required columns: {missing_columns}")

        result = merged_df.select(result_columns)

        context.log.info(f"Final merged data: {result.shape}")
        return result

    except Exception as e:
        context.log.error(f"Error merging data: {str(e)}")
        context.log.error(f"Exception type: {type(e).__name__}")
        context.log.error(f"Traceback: {traceback.format_exc()}")
        raise


@dg.asset(
    io_manager_key="io_manager",
    deps=["internal_sales_with_categories"],
    compute_kind="internal_preprocessing",
    group_name="preprocessing",
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

    try:
        context.log.info(f"Input data shape: {internal_sales_with_categories.shape}")

        # Check for required columns
        required_columns = ["SKU_NBR", "STORE_NBR", "CAT_DSC", "NEED_STATE", "TOTAL_SALES"]
        missing_columns = [
            col for col in required_columns if col not in internal_sales_with_categories.columns
        ]

        if missing_columns:
            context.log.error(f"Input data missing required columns: {missing_columns}")
            raise ValueError(f"Input data missing required columns: {missing_columns}")

        # Perform the normalization
        context.log.info("Calculating group counts and redistributing sales")
        result = (
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

        context.log.info(f"Normalized data shape: {result.shape}")
        return result

    except Exception as e:
        context.log.error(f"Error normalizing sales data: {str(e)}")
        context.log.error(f"Exception type: {type(e).__name__}")
        context.log.error(f"Traceback: {traceback.format_exc()}")
        raise


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

    try:
        context.log.info(f"Input data shape: {internal_normalized_sales_data.shape}")

        # Check for required columns
        required_columns = ["STORE_NBR", "CAT_DSC", "NEED_STATE", "TOTAL_SALES"]
        missing_columns = [
            col for col in required_columns if col not in internal_normalized_sales_data.columns
        ]

        if missing_columns:
            context.log.error(f"Input data missing required columns: {missing_columns}")
            raise ValueError(f"Input data missing required columns: {missing_columns}")

        # Get unique categories
        categories = (
            internal_normalized_sales_data.select(pl.col("CAT_DSC").unique()).to_series().to_list()
        )

        context.log.info(f"Found {len(categories)} unique categories: {categories}")

        # Create result dictionary
        result = {}

        # Process each category
        for cat in categories:
            context.log.info(f"Processing category: {cat}")

            # Filter data for this category
            cat_data = internal_normalized_sales_data.filter(pl.col("CAT_DSC") == cat)
            context.log.info(f"  Category {cat} has {cat_data.shape[0]} rows")

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
            context.log.info(f"  Found {len(need_states)} need states for category {cat}")

            # Pivot the data
            try:
                pivoted = merged.pivot(
                    index="STORE_NBR", values="Pct_of_Sales", columns="NEED_STATE"
                ).fill_null(0)

                # Create column rename mapping
                rename_map = {ns: f"% Sales {ns}" for ns in need_states if ns in pivoted.columns}

                # Rename columns and round values
                final_df = pivoted.rename(rename_map)

                # Round all percentage columns
                round_cols = [
                    f"% Sales {ns}" for ns in need_states if f"% Sales {ns}" in final_df.columns
                ]
                if round_cols:
                    final_df = final_df.with_columns([pl.col(col).round(2) for col in round_cols])

                # Add to result dictionary
                result[cat] = final_df
                context.log.info(f"  Successfully processed category {cat}: {final_df.shape}")

            except Exception as e:
                context.log.error(f"  Error pivoting data for category {cat}: {str(e)}")
                # Create a minimal dataframe with just STORE_NBR to avoid pipeline failure
                store_nums = store_sales.select("STORE_NBR")
                for ns in need_states:
                    store_nums = store_nums.with_columns(pl.lit(0.0).alias(f"% Sales {ns}"))
                result[cat] = store_nums
                context.log.info(
                    f"  Created fallback dataframe for category {cat}: {store_nums.shape}"
                )

        context.log.info(f"Successfully created category dictionary with {len(result)} categories")
        return result

    except Exception as e:
        context.log.error(f"Error creating sales by category: {str(e)}")
        context.log.error(f"Exception type: {type(e).__name__}")
        context.log.error(f"Traceback: {traceback.format_exc()}")
        raise


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

    try:
        context.log.info(f"Writing dictionary with {len(internal_sales_by_category)} categories")

        # Validate input
        if not internal_sales_by_category:
            context.log.error("Empty category dictionary")
            raise ValueError("Cannot write empty category dictionary")

        # Detailed logging for debugging
        for cat, df in internal_sales_by_category.items():
            context.log.info(f"Category {cat}: {df.shape} rows, columns: {df.columns}")

        # Write data
        context.log.info("Calling sales_by_category_writer.write()")
        context.resources.sales_by_category_writer.write(data=internal_sales_by_category)

        # Collect all unique store numbers across all categories
        all_stores = set()
        for category_df in internal_sales_by_category.values():
            stores = category_df.select("STORE_NBR").unique().to_series().to_list()
            all_stores.update(stores)

        context.log.info(f"Found {len(all_stores)} unique stores across all categories")

        # Add metadata
        context.add_output_metadata(
            metadata={
                "num_categories": len(internal_sales_by_category),
                "num_stores": len(all_stores),
            }
        )

        context.log.info("Successfully wrote sales by category data")

    except Exception as e:
        context.log.error(f"Error writing sales by category: {str(e)}")
        context.log.error(f"Exception type: {type(e).__name__}")
        context.log.error(f"Traceback: {traceback.format_exc()}")
        raise
