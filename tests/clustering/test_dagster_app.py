"""Tests for Dagster app entry point (app.py)."""

import pytest
import os
from unittest import mock

import dagster as dg

import src.clustering.dagster.app as app


def test_get_dagster_home_env(monkeypatch):
    monkeypatch.setenv("DAGSTER_HOME", "/tmp/test_dagster_home")
    assert app.get_dagster_home() == "/tmp/test_dagster_home"


def test_get_dagster_home_default(monkeypatch):
    monkeypatch.delenv("DAGSTER_HOME", raising=False)
    result = app.get_dagster_home()
    assert result.endswith("dagster_home")


def test_run_job_calls_execute(monkeypatch):
    # Patch defs.get_job_def and job.execute_in_process
    mock_job = mock.Mock()
    mock_job.execute_in_process.return_value = None
    monkeypatch.setattr(app.defs, "get_job_def", lambda name: mock_job)
    monkeypatch.setattr(dg.DagsterInstance, "get", lambda: mock.Mock(get_ref=lambda: None))
    app.run_job(env="dev")
    assert mock_job.execute_in_process.called


def test_run_app(monkeypatch):
    # Patch defs.get_job_def and job.execute_in_process
    mock_job = mock.Mock()
    mock_job.execute_in_process.return_value = None
    monkeypatch.setattr(app.defs, "get_job_def", lambda name: mock_job)
    monkeypatch.setattr(dg.DagsterInstance, "get", lambda: mock.Mock(get_ref=lambda: None))
    app.run_app(host="localhost", port=3000, env="dev")
    assert mock_job.execute_in_process.called


def test_run_job_invalid_job(monkeypatch):
    # Simulate get_job_def returning None
    monkeypatch.setattr(app.defs, "get_job_def", lambda name: None)
    monkeypatch.setattr(dg.DagsterInstance, "get", lambda: mock.Mock(get_ref=lambda: None))
    with pytest.raises(AttributeError):
        app.run_job(env="dev")


def test_run_job_env_var(monkeypatch):
    # Ensure DAGSTER_ENV is set
    monkeypatch.setenv("DAGSTER_ENV", "staging")
    mock_job = mock.Mock()
    mock_job.execute_in_process.return_value = None
    monkeypatch.setattr(app.defs, "get_job_def", lambda name: mock_job)
    monkeypatch.setattr(dg.DagsterInstance, "get", lambda: mock.Mock(get_ref=lambda: None))
    app.run_job(env="staging")
    assert os.environ["DAGSTER_ENV"] == "staging"


def test_run_app_port_and_host(monkeypatch):
    # Test that run_app accepts custom host/port
    mock_job = mock.Mock()
    mock_job.execute_in_process.return_value = None
    monkeypatch.setattr(app.defs, "get_job_def", lambda name: mock_job)
    monkeypatch.setattr(dg.DagsterInstance, "get", lambda: mock.Mock(get_ref=lambda: None))
    app.run_app(host="127.0.0.1", port=1234, env="dev")
    assert mock_job.execute_in_process.called
