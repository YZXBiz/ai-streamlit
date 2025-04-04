"""Data IO resources for Dagster pipelines."""

import dagster as dg

from clustering.io import (
    BlobReader,
    BlobWriter,
    CSVReader,
    CSVWriter,
    ExcelReader,
    ExcelWriter,
    ParquetReader,
    ParquetWriter,
    PickleReader,
    PickleWriter,
    Reader,
    SnowflakeReader,
    SnowflakeWriter,
    Writer,
)


@dg.resource(
    config_schema={
        "kind": dg.Field(dg.String, is_required=True, description="Type of reader"),
        "config": dg.Field(
            dg.Permissive(),
            is_required=True,
            description="Configuration for the reader",
        ),
    }
)
def data_reader(context: dg.InitResourceContext) -> Reader:
    """Resource for reading data.

    Args:
        context: The context for initializing the resource.

    Returns:
        Reader: A configured reader.
    """
    kind = context.resource_config["kind"]
    config = context.resource_config["config"]

    # Create reader based on kind
    reader_map = {
        "ParquetReader": ParquetReader,
        "CSVReader": CSVReader,
        "ExcelReader": ExcelReader,
        "PickleReader": PickleReader,
        "SnowflakeReader": SnowflakeReader,
        "BlobReader": BlobReader,
    }

    # Check if requested reader exists in our map
    reader_cls = reader_map.get(kind)

    if not reader_cls:
        raise ValueError(f"Unknown reader kind: {kind}")

    return reader_cls(**config)


@dg.resource(
    config_schema={
        "kind": dg.Field(dg.String, is_required=True, description="Type of writer"),
        "config": dg.Field(
            dg.Permissive(),
            is_required=True,
            description="Configuration for the writer",
        ),
    }
)
def data_writer(context: dg.InitResourceContext) -> Writer:
    """Resource for writing data.

    Args:
        context: The context for initializing the resource.

    Returns:
        Writer: A configured writer.
    """
    kind = context.resource_config["kind"]
    config = context.resource_config["config"]

    # Create writer based on kind
    writer_map = {
        "ParquetWriter": ParquetWriter,
        "CSVWriter": CSVWriter,
        "ExcelWriter": ExcelWriter,
        "PickleWriter": PickleWriter,
        "SnowflakeWriter": SnowflakeWriter,
        "BlobWriter": BlobWriter,
    }

    # Check if requested writer exists in our map
    writer_cls = writer_map.get(kind)

    if not writer_cls:
        raise ValueError(f"Unknown writer kind: {kind}")

    return writer_cls(**config)
