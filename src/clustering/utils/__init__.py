"""Utility functions for the clustering package."""

# Data processing utilities
# Make these utilities available as helpers.<func> for backward compatibility
import sys
from types import ModuleType

# For backward compatibility - expose everything through helpers
from clustering.utils.data_processing import *
from clustering.utils.data_processing import (
    clean_ns,
    create_cat_dict,
    distribute_sales_evenly,
    merge_dataframes,
    merge_sales_ns,
)

# DVC utilities
from clustering.utils.dvc import *
from clustering.utils.dvc import handle_dvc_lineage

# Merging utilities
from clustering.utils.merging import *
from clustering.utils.merging import merge_int_ext

# Create a virtual module for backwards compatibility
helpers = ModuleType("helpers")
sys.modules["clustering.utils.helpers"] = helpers

# Copy all functions to the helpers module
for func_name in dir():
    if not func_name.startswith("_") and func_name not in ["sys", "ModuleType", "helpers"]:
        setattr(helpers, func_name, globals()[func_name])

__all__ = [
    # Data processing
    "clean_ns",
    "create_cat_dict",
    "distribute_sales_evenly",
    "merge_dataframes",
    "merge_sales_ns",
    # DVC
    "handle_dvc_lineage",
    # Merging
    "merge_int_ext",
    # Legacy
    "helpers",
]
