"""Utilities for DVC and version control operations."""

import os
from typing import Any, Optional


def handle_dvc_lineage(
    folder_path: str,
    commit_message: str,
    remote_name: str = "clusteringremote",
    push_to_remote: bool = False,
    logger: Optional[Any] = None,
) -> None:
    """Handle DVC lineage for data versioning.

    Tracks a folder/file with DVC, commits to Git, and optionally pushes to remote.

    This function:
    - Adds the folder to DVC tracking
    - Adds the generated DVC file to Git
    - Commits the changes
    - Optionally pushes data to remote storage

    Args:
        folder_path: Path to folder/file to track
        commit_message: Git commit message
        remote_name: DVC remote name (default: "clusteringremote")
        push_to_remote: Whether to push data to remote (default: False)
        logger: Optional logger to use for logging
    """
    if logger:
        logger.info(f"Adding {folder_path} to DVC tracking")

    # Add to DVC
    os.system(f"dvc add {folder_path}")

    if logger:
        logger.info(f"Adding {folder_path}.dvc to Git")

    # Add to Git
    os.system(f"git add {folder_path}.dvc .gitignore")

    if logger:
        logger.info(f"Committing with message: {commit_message}")

    # Commit changes
    os.system(f"git commit -m '{commit_message}'")

    # Push to remote if requested
    if push_to_remote:
        if logger:
            logger.info(f"Pushing data to remote {remote_name}")
        os.system(f"dvc push -r {remote_name}")
