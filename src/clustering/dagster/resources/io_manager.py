"""IO manager for the clustering pipeline."""

import os
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import dagster as dg
import polars as pl
from dagster import io_manager

from clustering.config import SETTINGS


class ClusteringIOManagerConfig:
    """Configuration for the clustering IO manager."""

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the IO manager configuration.

        Args:
            base_dir: Base directory for storing outputs
        """
        self.base_dir = base_dir or os.path.join(SETTINGS.ROOT_DIR, "outputs/dagster_storage")


class ClusteringIOManager:
    """IO manager for clustering pipeline assets.

    This IO manager handles storage of intermediate and final outputs from
    the clustering pipeline, supporting various data types.
    """

    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the IO manager.

        Args:
            base_dir: Base directory for storing outputs
        """
        self.config = ClusteringIOManagerConfig(base_dir)
        os.makedirs(self.config.base_dir, exist_ok=True)

    def _get_path(self, context: Union[dg.OutputContext, dg.InputContext]) -> Path:
        """Get the path for an asset.

        Args:
            context: The context containing asset information

        Returns:
            Path to the asset file
        """
        asset_key_segments = context.asset_key.path
        path = Path(self.config.base_dir)

        # Create directories based on asset key path
        for segment in asset_key_segments[:-1]:
            path = path / segment
            os.makedirs(path, exist_ok=True)

        # Determine file extension based on data type
        filename = f"{asset_key_segments[-1]}"

        # Add the partition info to the path if it exists
        if hasattr(context, "partition") and context.partition is not None:
            path = path / str(context.partition)
            os.makedirs(path, exist_ok=True)

        # Return full path with filename
        return path / filename

    def _get_format(self, obj: Any) -> str:
        """Get the appropriate format for an object.

        Args:
            obj: The object to determine the format for

        Returns:
            A string representing the format
        """
        if isinstance(obj, pl.DataFrame):
            return "parquet"
        elif isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            # Dictionary with string keys - check values
            values = list(obj.values())
            if values and all(isinstance(v, pl.DataFrame) for v in values):
                return "parquet_dict"
        # Default format for other types
        return "pickle"

    def handle_output(self, context: dg.OutputContext, obj: Any) -> None:
        """Handle storage of an output.

        Args:
            context: The context containing output information
            obj: The object to store
        """
        if obj is None:
            return

        path = self._get_path(context)
        os.makedirs(path.parent, exist_ok=True)

        # Determine format and save accordingly
        format_type = self._get_format(obj)

        if format_type == "parquet":
            df = obj
            parquet_path = f"{path}.parquet"
            df.write_parquet(parquet_path)
            context.log.info(f"Saved DataFrame to {parquet_path}")

        elif format_type == "parquet_dict":
            # Save each DataFrame in the dictionary
            dict_dir = path
            os.makedirs(dict_dir, exist_ok=True)

            # Save dictionary keys to a manifest file
            with open(f"{dict_dir}/_manifest.txt", "w") as f:
                f.write("\n".join(obj.keys()))

            # Save each DataFrame
            for key, df in obj.items():
                df_path = f"{dict_dir}/{key}.parquet"
                df.write_parquet(df_path)

            context.log.info(f"Saved dictionary of DataFrames to {dict_dir}")

        else:
            # Use pickle for other object types
            pickle_path = f"{path}.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(obj, f)
            context.log.info(f"Saved object to {pickle_path}")

    def load_input(self, context: dg.InputContext) -> Any:
        """Load input from storage.

        Args:
            context: The context containing input information

        Returns:
            The loaded object
        """
        path = self._get_path(context)

        # Try different formats
        parquet_path = f"{path}.parquet"
        pickle_path = f"{path}.pkl"
        dict_dir = path

        if os.path.exists(parquet_path):
            # Load DataFrame from parquet
            df = pl.read_parquet(parquet_path)
            return df

        elif os.path.exists(pickle_path):
            # Load object from pickle
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        elif os.path.exists(dict_dir) and os.path.isdir(dict_dir):
            # Check for dictionary of DataFrames
            manifest_path = f"{dict_dir}/_manifest.txt"
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    keys = f.read().splitlines()

                # Load each DataFrame
                result = {}
                for key in keys:
                    df_path = f"{dict_dir}/{key}.parquet"
                    if os.path.exists(df_path):
                        result[key] = pl.read_parquet(df_path)

                return result

        # If we get here, the input doesn't exist
        raise ValueError(f"Input not found at {path}")


@io_manager
def clustering_io_manager(init_context: dg.InitResourceContext) -> ClusteringIOManager:
    """Factory function for clustering IO manager.

    Args:
        init_context: The context for initializing the resource

    Returns:
        A configured IO manager
    """
    # Get base directory from config if provided
    base_dir = getattr(init_context.resource_config, "base_dir", None)
    return ClusteringIOManager(base_dir)
