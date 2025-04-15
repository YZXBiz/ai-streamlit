"""Tests for Dagster pipeline definitions (definitions.py)."""

import pytest
import dagster as dg

import src.clustering.dagster.definitions as definitions


def test_defs_is_definitions():
    assert hasattr(definitions, "defs")
    assert isinstance(definitions.defs, dg.Definitions)


def test_create_definitions_returns_definitions():
    defs = definitions.create_definitions(env="dev")
    assert isinstance(defs, dg.Definitions)


def test_get_all_jobs_and_assets():
    defs = definitions.create_definitions(env="dev")
    jobs = defs.get_all_jobs()
    assets = defs.get_all_assets()
    assert isinstance(jobs, list)
    assert isinstance(assets, list)
    assert len(jobs) > 0
    assert len(assets) > 0


def test_create_definitions_invalid_env():
    # Should still return a Definitions object or raise a clear error
    try:
        defs = definitions.create_definitions(env="nonexistent")
        assert isinstance(defs, dg.Definitions)
    except Exception as e:
        assert "config" in str(e).lower() or "not found" in str(e).lower()


def test_job_selection_and_tags():
    defs = definitions.create_definitions(env="dev")
    jobs = defs.get_all_jobs()
    for job in jobs:
        assert hasattr(job, "name")
        assert hasattr(job, "tags")
        assert isinstance(job.tags, dict)


def test_asset_selection_and_metadata():
    defs = definitions.create_definitions(env="dev")
    assets = defs.get_all_assets()
    for asset in assets:
        assert hasattr(asset, "key")
        assert hasattr(asset, "metadata")
        assert isinstance(asset.metadata, dict)
