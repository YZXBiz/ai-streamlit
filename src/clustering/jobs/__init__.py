"""Jobs for clustering operations.

This package provides job classes for different clustering operations,
as well as utilities for job execution.
"""

# Job type union
from typing import Union

# Base job class
from clustering.jobs.base import Job, Locals
from clustering.jobs.external_clustering import ExternalTrainingJob
from clustering.jobs.external_preprocessing import ExternalPreproJob
from clustering.jobs.internal_clustering import InternalTrainingJob

# Concrete job implementations
from clustering.jobs.internal_preprocessing import InternalPreproJob
from clustering.jobs.merge_int_ext import MergeIntExtJob

# Job utilities
from clustering.jobs.utils import get_path_safely, track_dvc_lineage, validate_dataframe

JobKind = Union[
    InternalPreproJob,
    InternalTrainingJob,
    ExternalPreproJob,
    ExternalTrainingJob,
    MergeIntExtJob,
]

__all__ = [
    # Base classes
    "Job",
    "Locals",
    # Utilities
    "get_path_safely",
    "track_dvc_lineage",
    "validate_dataframe",
    # Concrete job implementations
    "InternalPreproJob",
    "InternalTrainingJob",
    "ExternalPreproJob",
    "ExternalTrainingJob",
    "MergeIntExtJob",
    # Type definitions
    "JobKind",
]
