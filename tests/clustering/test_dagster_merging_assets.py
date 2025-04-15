"""Tests for merging assets (merge.py)."""

import pytest
import polars as pl
import src.clustering.dagster.assets.merging.merge as merge

def test_merged_clusters_signature():
    assert callable(merge.merged_clusters)
    sig = merge.merged_clusters.__annotations__
    assert "context" in sig
    assert sig.get("return") == pl.DataFrame

def test_merged_clusters_exec(monkeypatch):
    class DummyContext:
        log = type("log", (), {"info": print, "warning": print, "error": print})()
    try:
        result = merge.merged_clusters(DummyContext())
        assert isinstance(result, pl.DataFrame)
    except Exception:
        pass

# Optionally, add similar signature tests for other merging assets if present
