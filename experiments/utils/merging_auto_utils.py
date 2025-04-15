#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merging_auto_utils.py

Helper functions for the merging_auto_main pipeline, including:
  - Name sanitization
  - Locating the newest run folders
  - Matching cluster-labeled CSVs by prefix
  - Creating a time-stamped "Merged_Clustering_Output_Run_{YYYYMMDD_HHMM}" folder
"""

import sys
import os
import logging
import re
from datetime import datetime

import pandas as pd


# ----------------------------------------------------------------------
# Helper function to sanitize category names (to match filenames)
# ----------------------------------------------------------------------
def sanitize_name(name: str) -> str:
    """
    Convert a category name into a form that matches how our
    'df_clustered_*.csv' files are named.
    For example:
      - "DIET/NUTRITION" => "DIET_NUTRITION"
      - "FLOWERS/PRODUCE" => "FLOWERS_PRODUCE"
      - "HOUSEHOLD PAPER" => "HOUSEHOLD_PAPER"
      - "HAND & BODY" => "HAND_&_BODY"

    We preserve the & symbol, replace spaces/slashes/dashes with underscores,
    and remove anything else non-alphanumeric (besides underscores and &).
    """
    import re

    # Lowercase the input
    name = name.lower()
    # Replace slashes with underscores
    name = re.sub(r"/", "_", name)
    # Replace spaces with underscores
    name = re.sub(r"\s+", "_", name)
    # Replace dashes with underscores
    name = re.sub(r"-", "_", name)
    # Preserve ampersand, remove anything else not alphanumeric or underscore
    name = re.sub(r"[^a-z0-9_&]", "", name)
    # Finally, uppercase the result
    return name.upper()


# ----------------------------------------------------------------------
# Utility functions from merging_auto_utils
# ----------------------------------------------------------------------
def get_latest_run_folder(base_path: str, run_prefix: str) -> str:
    """
    Finds the most recent subfolder in `base_path` that starts with `run_prefix`.
    Returns the subfolder name or None if none found.
    """
    if not os.path.isdir(base_path):
        return None

    folders = [f for f in os.listdir(base_path) if f.startswith(run_prefix)]
    if not folders:
        return None

    # Sort by date/time in the folder name (assuming standard naming with _YYYYMMDD_HHMM)
    folders_sorted = sorted(folders, reverse=True)
    return folders_sorted[0]  # the latest


def get_clustered_file(folder_path: str, sanitized_keyword: str) -> str:
    """
    Given a folder_path and a sanitized keyword (e.g. "VITAMINS"),
    returns the path to the CSV that matches the pattern
    'df_clustered_{sanitized_keyword}_*.csv'.
    If multiple CSVs match, it picks the most recent (by sorted name).
    If none found, returns None.
    """
    if not os.path.isdir(folder_path):
        return None

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    # e.g. df_clustered_VITAMINS_20250101_1200.csv
    pattern = f"df_clustered_{sanitized_keyword}_"
    matches = [f for f in all_files if f.startswith(pattern)]

    if not matches:
        return None

    # Sort matches in descending order (so the most recent is first, if they differ by timestamp)
    matches_sorted = sorted(matches, reverse=True)
    chosen = matches_sorted[0]
    return os.path.join(folder_path, chosen)


def create_merged_output_folder(base_merged_path: str):
    """
    Creates and returns a new folder named 'Merged_Clustering_Output_Run_{YYYYMMDD_HHMM}'.
    Also returns the run timestamp used in the folder name.
    """
    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"Merged_Clustering_Output_Run_{run_ts}"
    out_path = os.path.join(base_merged_path, folder_name)
    os.makedirs(out_path, exist_ok=True)
    return out_path, run_ts


def create_merged_output_folder_all(base_merged_path: str):
    """
    Creates and returns a new folder named 'Merged_Clustering_Output_Run_All_{YYYYMMDD_HHMM}'.
    Also returns the run timestamp used in the folder name.
    """
    run_ts = datetime.now().strftime("%Y%m%d_%H%M")
    folder_name = f"Merged_Clustering_Output_Run_All_{run_ts}"
    out_path = os.path.join(base_merged_path, folder_name)
    os.makedirs(out_path, exist_ok=True)
    return out_path, run_ts
