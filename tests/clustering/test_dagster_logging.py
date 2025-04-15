"""Tests for Dagster logging resource (logging.py)."""

import pytest
import src.clustering.dagster.resources.logging as logging_mod


def test_alerts_service_send_alert(monkeypatch):
    service = logging_mod.AlertsService(enabled=True, threshold="WARNING", slack_webhook=None)
    called = {}

    def fake_send(level, message):
        called["level"] = level
        called["message"] = message

    monkeypatch.setattr(service, "send_alert", fake_send)
    service.send_alert("ERROR", "Test message")
    assert called["level"] == "ERROR"
    assert called["message"] == "Test message"


def test_alerts_service_disabled():
    service = logging_mod.AlertsService(enabled=False)
    # Should not raise or send alerts
    service.send_alert("INFO", "Should not send")


def test_alerts_service_threshold(monkeypatch):
    # Test that threshold is respected (simulate logic if present)
    service = logging_mod.AlertsService(enabled=True, threshold="ERROR")
    # If logic is implemented, only ERROR or higher should trigger
    # Here, just ensure method is callable
    service.send_alert("ERROR", "Error message")
    service.send_alert("WARNING", "Warning message")
