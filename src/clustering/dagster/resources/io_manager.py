"""IO manager for the clustering pipeline."""

import os
import pickle
from pathlib import Path
from typing import Any

import dagster as dg
from dagster import InputContext, IOManager, OutputContext

# Try to import polars for DataFrame handling
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# Try to import pandas for DataFrame handling
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class EnhancedFileSystemIOManager(IOManager):
    """An enhanced IO manager that efficiently handles various data types."""

    def __init__(self, base_dir: str):
        """Initialize the IO manager.

        Args:
            base_dir: Base directory for storing outputs
        """
        self.base_dir = Path(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _get_path(self, context: InputContext | OutputContext, extension: str = None) -> Path:
        """Get the path for an asset.

        Args:
            context: The input or output context
            extension: Optional file extension override

        Returns:
            Path to the asset file
        """
        # Use asset key as filename, or metadata key if specified
        if hasattr(context, "metadata") and context.metadata and "filename" in context.metadata:
            filename = context.metadata["filename"]
        else:
            asset_key = context.asset_key.path[-1] if context.asset_key else "default"
            filename = asset_key

        # If no extension is provided, use .pkl as default
        if not extension:
            return self.base_dir / f"{filename}.pkl"
        return self.base_dir / f"{filename}{extension}"

    def _get_extension_for_obj(self, obj: Any) -> str:
        """Determine the best file extension based on object type.

        Args:
            obj: The object to store

        Returns:
            Appropriate file extension
        """
        if POLARS_AVAILABLE and isinstance(obj, pl.DataFrame):
            return ".parquet"
        elif PANDAS_AVAILABLE and isinstance(obj, pd.DataFrame):
            return ".parquet"
        else:
            return ".pkl"

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Handle an output from a compute function.

        Args:
            context: The output context
            obj: The output value
        """
        extension = self._get_extension_for_obj(obj)
        path = self._get_path(context, extension)

        # Handle different types of objects efficiently
        if POLARS_AVAILABLE and isinstance(obj, pl.DataFrame):
            obj.write_parquet(path)
        elif PANDAS_AVAILABLE and isinstance(obj, pd.DataFrame):
            obj.to_parquet(path)
        else:
            # For all other types, use pickle
            with open(path, "wb") as f:
                pickle.dump(obj, f)

        # Log the saved location
        context.log.info(f"Saved output to {path}")

    def load_input(self, context: InputContext) -> Any:
        """Load an input for a compute function.

        Args:
            context: The input context

        Returns:
            The input value
        """
        # First try .parquet extension
        parquet_path = self._get_path(context, ".parquet")
        if parquet_path.exists():
            context.log.info(f"Loading Parquet data from {parquet_path}")
            if POLARS_AVAILABLE:
                return pl.read_parquet(parquet_path)
            elif PANDAS_AVAILABLE:
                return pd.read_parquet(parquet_path)

        # Fall back to pickle
        pkl_path = self._get_path(context, ".pkl")
        if pkl_path.exists():
            context.log.info(f"Loading pickled data from {pkl_path}")
            with open(pkl_path, "rb") as f:
                return pickle.load(f)

        # If we reach here, neither file exists
        paths_tried = [parquet_path, pkl_path]
        raise FileNotFoundError(f"Could not find input file at any of these paths: {paths_tried}")


@dg.io_manager(
    config_schema={
        "base_dir": dg.Field(
            dg.String,
            default_value="outputs",
            description="Base directory for storing outputs",
        ),
    }
)
def clustering_io_manager(init_context: dg.InitResourceContext) -> IOManager:
    """Factory function for the clustering IO manager.

    Args:
        init_context: The context for initializing the resource

    Returns:
        A configured IO manager
    """
    # Get configuration
    config = init_context.resource_config
    base_dir = config.get("base_dir", "outputs")

    # Create the IO manager
    return EnhancedFileSystemIOManager(base_dir=base_dir)
