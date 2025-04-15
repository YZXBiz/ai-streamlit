"""Tests for Dagster data IO resources (data_io.py)."""

import pytest
from dagster import build_init_resource_context

import src.clustering.dagster.resources.data_io as data_io


def test_data_reader_resource_minimal(monkeypatch):
    class DummyReader:
        pass

    monkeypatch.setattr(data_io.io_module, "Reader", DummyReader)
    context = build_init_resource_context(config={"kind": "DummyReader", "config": {}})
    # Should not raise
    result = data_io.data_reader(context)
    assert isinstance(result, DummyReader)


def test_data_writer_resource_minimal(monkeypatch):
    class DummyWriter:
        pass

    monkeypatch.setattr(data_io.io_module, "Writer", DummyWriter)
    context = build_init_resource_context(config={"kind": "DummyWriter", "config": {}})
    # Should not raise
    result = data_io.data_writer(context)
    assert isinstance(result, DummyWriter)


def test_data_reader_invalid_kind(monkeypatch):
    # Simulate missing reader kind
    context = build_init_resource_context(config={"kind": "NonExistentReader", "config": {}})
    with pytest.raises(Exception):
        data_io.data_reader(context)


def test_data_writer_invalid_kind(monkeypatch):
    # Simulate missing writer kind
    context = build_init_resource_context(config={"kind": "NonExistentWriter", "config": {}})
    with pytest.raises(Exception):
        data_io.data_writer(context)


def test_data_reader_config_validation(monkeypatch):
    # Simulate missing config
    context = build_init_resource_context(config={"kind": "DummyReader"})
    with pytest.raises(Exception):
        data_io.data_reader(context)


def test_data_writer_config_validation(monkeypatch):
    # Simulate missing config
    context = build_init_resource_context(config={"kind": "DummyWriter"})
    with pytest.raises(Exception):
        data_io.data_writer(context)
