"""External preprocessing assets for the clustering pipeline."""

import dagster as dg
import polars as pl

from clustering.utils.helpers import merge_dataframes


@dg.asset(
    io_manager_key="io_manager",
    compute_kind="external_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"input_external_sales_reader"},
)
def external_features_data(
    context: dg.AssetExecutionContext,
) -> pl.DataFrame:
    """Load and merge external feature data.

    Args:
        context: Asset execution context

    Returns:
        DataFrame containing merged external data
    """
    context.log.info("Reading external features data")

    input_external_readers = context.resources.input_external_sales_reader

    # Ensure readers are in a list format
    if not isinstance(input_external_readers, list):
        input_external_readers = [input_external_readers]

    # Read and convert all external data to Polars
    df_list: list[pl.DataFrame] = []
    for input_data in input_external_readers:
        context.log.info(f"Reading external feature: {input_data}")
        data = input_data.read()

        if not isinstance(data, pl.DataFrame):
            data = pl.from_pandas(data)

        df_list.append(data)

    # Merge all dataframes
    context.log.info("Merging all external feature dataframes")
    if len(df_list) == 1:
        return df_list[0]

    # Convert to pandas for merging, then back to polars
    pandas_df_list = [df.to_pandas() for df in df_list]
    merged_pandas = merge_dataframes(pandas_df_list)

    return pl.from_pandas(merged_pandas)


@dg.asset(
    io_manager_key="io_manager",
    deps=["external_features_data"],
    compute_kind="external_preprocessing",
    group_name="preprocessing",
    required_resource_keys={"output_external_data_writer"},
)
def preprocessed_external_data(
    context: dg.AssetExecutionContext,
    external_features_data: pl.DataFrame,
) -> pl.DataFrame:
    """Process external data and save the results.

    Args:
        context: Asset execution context
        external_features_data: External features data

    Returns:
        Processed external data
    """
    context.log.info("Processing external data")

    # No additional processing needed in this simple case
    # Add any processing steps here as needed

    context.log.info("Saving preprocessed external data")
    output_data_writer = context.resources.output_external_data_writer

    # Write the data in the appropriate format
    if hasattr(output_data_writer, "requires_pandas") and output_data_writer.requires_pandas:
        output_data_writer.write(data=external_features_data.to_pandas())
    else:
        output_data_writer.write(data=external_features_data)

    return external_features_data
