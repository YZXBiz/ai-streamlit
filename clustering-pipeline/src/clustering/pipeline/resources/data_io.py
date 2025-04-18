"""Data IO resources for Dagster pipelines."""

import inspect

import dagster as dg

from shared.io import Reader, Writer
import shared.io.readers as readers
import shared.io.writers as writers


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

    # Dynamically build reader map from io module
    reader_map = {
        name: cls
        for name, cls in inspect.getmembers(readers)
        if inspect.isclass(cls) and issubclass(cls, Reader) and cls is not Reader
    }

    # Check if requested reader exists in our map
    reader_cls = reader_map.get(kind)

    if not reader_cls:
        raise ValueError(f"Unknown reader kind: {kind}, available readers: {reader_map.keys()}")

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

    # Dynamically build writer map from io module
    writer_map = {
        name: cls
        for name, cls in inspect.getmembers(writers)
        if inspect.isclass(cls) and issubclass(cls, Writer) and cls is not Writer
    }

    # Check if requested writer exists in our map
    writer_cls = writer_map.get(kind)

    if not writer_cls:
        raise ValueError(f"Unknown writer kind: {kind}, available writers: {writer_map.keys()}")

    return writer_cls(**config)
