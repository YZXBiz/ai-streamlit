"""Utilities for job operations and shared functionality."""

from typing import Any, Optional, Type, Union, cast

import polars as pl

from clustering.io import datasets
from clustering.utils.dvc import handle_dvc_lineage


def get_path_safely(obj: Any) -> str:
    """Get path from an object, handling cases where path doesn't exist.

    Args:
        obj: Object that might have a path attribute

    Returns:
        Path string or string representation of the object
    """
    return cast(str, getattr(obj, "path", str(obj)))


def validate_dataframe(data: pl.DataFrame, schema: Type[Any], logger: Optional[Any] = None) -> pl.DataFrame:
    """Validate a dataframe against a schema.

    Args:
        data: Polars dataframe to validate
        schema: Schema class with a check method
        logger: Optional logger for logging

    Returns:
        Validated dataframe
    """
    if logger:
        logger.info(f"Validating data with schema: {schema.__name__}")

    # Schema check may expect pandas DataFrame, so we'll convert if needed
    if hasattr(schema, "requires_pandas") and schema.requires_pandas:
        # Convert to pandas for validation
        pandas_df = data.to_pandas()
        validated_pandas = cast(Any, schema.check(data=pandas_df))
        # Convert back to polars
        return pl.from_pandas(validated_pandas)
    else:
        # Assume schema can handle polars directly
        return cast(pl.DataFrame, schema.check(data=data))


def track_dvc_lineage(
    io_object: Union[datasets.ReaderKind, datasets.WriterKind],
    commit_message: str,
    push_to_remote: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """Track DVC lineage for a reader or writer object.

    Provides a safer way to track DVC lineage by handling path resolution.

    Args:
        io_object: Reader or writer object with path attribute
        commit_message: Commit message for Git
        push_to_remote: Whether to push to remote (default: False)
        logger: Optional logger for logging
    """
    path = get_path_safely(io_object)
    handle_dvc_lineage(folder_path=path, commit_message=commit_message, push_to_remote=push_to_remote, logger=logger)
