"""External preprocessing assets for the clustering pipeline."""

import dagster as dg
import polars as pl

from clustering.utils.helpers import merge_dataframes


@dg.asset(
    io_manager_key="io_manager",
    key_prefix="raw",
    compute_kind="external_preprocessing",
    group_name="preprocessing",
)
def external_features_data(
    context: dg.AssetExecutionContext,
    input_external_readers=dg.ResourceParam(dg.InitResourceContext),
) -> pl.DataFrame:
    """Load and merge external feature data.

    Args:
        context: Asset execution context
        input_external_readers: List of readers for external data

    Returns:
        DataFrame containing merged external data
    """
    context.log.info("Reading external features data")

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
    deps=["raw/external_features_data"],
    compute_kind="external_preprocessing",
    group_name="preprocessing",
)
def preprocessed_external_data(
    context: dg.AssetExecutionContext,
    external_features_data: pl.DataFrame,
    output_data_writer=dg.ResourceParam(dg.InitResourceContext),
) -> pl.DataFrame:
    """Process external data and save the results.

    Args:
        context: Asset execution context
        external_features_data: External features data
        output_data_writer: Writer for output data

    Returns:
        Processed external data
    """
    context.log.info("Processing external data")

    # In this simple case, there's no additional processing needed
    # We just save the merged data
    # Add any additional processing steps here as needed

    # Save the data
    context.log.info("Saving preprocessed external data")

    # Check if the writer expects pandas DataFrame
    if hasattr(output_data_writer, "requires_pandas") and output_data_writer.requires_pandas:
        output_data_writer.write(data=external_features_data.to_pandas())
    else:
        output_data_writer.write(data=external_features_data)

    # Return the processed data for downstream assets
    return external_features_data
