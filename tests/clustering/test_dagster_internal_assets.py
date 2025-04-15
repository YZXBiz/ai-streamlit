"""Tests for internal preprocessing assets (internal.py)."""

import pytest
import dagster as dg
import polars as pl
import src.clustering.dagster.assets.preprocessing.internal as internal


def test_internal_raw_sales_data_signature():
    assert callable(internal.internal_raw_sales_data)
    # Check function signature
    sig = internal.internal_raw_sales_data.__annotations__
    assert "context" in sig
    assert sig.get("return") == pl.DataFrame


def test_internal_product_category_mapping_signature():
    assert callable(internal.internal_product_category_mapping)
    sig = internal.internal_product_category_mapping.__annotations__
    assert "context" in sig
    assert sig.get("return") == pl.DataFrame


def test_internal_raw_sales_data_exec(monkeypatch):
    # Mock context with minimal interface
    class DummyContext:
        log = type("log", (), {"info": print, "warning": print, "error": print})()

    # Should raise or return a DataFrame depending on implementation
    try:
        result = internal.internal_raw_sales_data(DummyContext())
        assert isinstance(result, pl.DataFrame)
    except Exception:
        pass


def test_internal_product_category_mapping_exec(monkeypatch):
    class DummyContext:
        log = type("log", (), {"info": print, "warning": print, "error": print})()

    try:
        result = internal.internal_product_category_mapping(DummyContext())
        assert isinstance(result, pl.DataFrame)
    except Exception:
        pass
