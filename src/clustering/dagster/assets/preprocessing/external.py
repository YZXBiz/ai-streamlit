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

    # Get reader from resources
    input_external_readers = context.resources.input_external_sales_reader

    # Check if input_external_readers is a list
    if not isinstance(input_external_readers, list):
        input_external_readers = [input_external_readers]

    # Read all external data
    df_list = []
    for input_data in input_external_readers:
        context.log.info(f"Reading external feature: {input_data}")
        data = input_data.read()

        # Convert to Polars if needed
        if not isinstance(data, pl.DataFrame):
            data = pl.from_pandas(data)

        df_list.append(data)

    # Merge all dataframes
    context.log.info("Merging all external feature dataframes")
    if len(df_list) == 1:
        merged_data = df_list[0]
    else:
        # Convert to pandas for the merge_dataframes function, then back to polars
        pandas_df_list = [df.to_pandas() for df in df_list]
        merged_pandas = merge_dataframes(pandas_df_list)
        merged_data = pl.from_pandas(merged_pandas)

    return merged_data


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

    # In this simple case, there's no additional processing needed
    # We just save the merged data
    # Add any additional processing steps here as needed

    # Save the data
    context.log.info("Saving preprocessed external data")

    # Get writer from resources
    output_data_writer = context.resources.output_external_data_writer

    # Check if the writer expects pandas DataFrame
    if hasattr(output_data_writer, "requires_pandas") and output_data_writer.requires_pandas:
        output_data_writer.write(data=external_features_data.to_pandas())
    else:
        output_data_writer.write(data=external_features_data)

    # Return the processed data for downstream assets
    return external_features_data
