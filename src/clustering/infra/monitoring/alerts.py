"""Alert service for the clustering pipeline.

This is a Pydantic-based implementation for alerts that supports multiple channels.
"""

import json
import logging
from typing import Any

import requests
from pydantic import BaseModel, Field


class AlertConfig(BaseModel):
    """Configuration for the alerting service."""

    enabled: bool = Field(True, description="Whether alerting is enabled")
    threshold: str = Field("ERROR", description="Minimum log level to trigger alerts")
    channels: list[str] = Field(["email"], description="Alert channels to use")
    slack_webhook: str | None = Field(None, description="Slack webhook URL")
    email_recipients: list[str] = Field(
        ["Jackson.Yang@cvshealth.com"], description="Email recipients"
    )
    alertmanager_url: str = Field(
        "http://localhost:9093", description="Prometheus Alertmanager URL"
    )

    @property
    def threshold_level(self) -> int:
        """Get the numeric log level threshold.

        If the threshold is not a valid log level, use ERROR as the default.
        """
        return getattr(logging, self.threshold, logging.ERROR)


class AlertingService:
    """Service for sending alerts from pipelines."""

    def __init__(
        self,
        enabled: bool,
        threshold: str,
        channels: list[str],
        slack_webhook: str | None,
        email_recipients: list[str],
        alertmanager_url: str,
        logger: Any,
    ):
        """Initialize the alerting service.

        Args:
            enabled: Whether alerting is enabled
            threshold: Minimum log level to trigger alerts
            channels: Alert channels to use
            slack_webhook: Slack webhook URL
            email_recipients: Email recipients
            alertmanager_url: Prometheus Alertmanager URL
            logger: Logger
        """
        self.enabled = enabled
        self.threshold = threshold
        self.threshold_level = getattr(logging, threshold, logging.ERROR)
        self.channels = channels
        self.slack_webhook = slack_webhook
        self.email_recipients = email_recipients
        self.alertmanager_url = alertmanager_url
        self.logger = logger

    def start(self) -> None:
        """Start the alerting service."""
        # Nothing to do on start
        pass

    def stop(self) -> None:
        """Stop the alerting service."""
        # Nothing to do on stop
        pass

    def alert(
        self,
        message: str,
        level: str = "ERROR",
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Send an alert.

        Args:
            message: The alert message
            level: Log level of the alert
            context: Additional context for the alert

        Returns:
            bool: Whether the alert was sent successfully
        """
        if not self.enabled:
            self.logger.info("Alerting disabled, would have sent: %s", message)
            return False

        level_num = getattr(logging, level, logging.ERROR)
        if level_num < self.threshold_level:
            self.logger.debug(
                "Alert level %s below threshold %s, not sending", level, self.threshold
            )
            return False

        context = context or {}
        success = False

        # Prepare metadata once
        alert_metadata = {
            "level": level,
            "message": message,
            **context,
        }

        # Send to configured channels
        for channel in self.channels:
            try:
                if channel == "slack" and self.slack_webhook:
                    success = self._send_slack_alert(message, alert_metadata)
                elif channel == "email" and self.email_recipients:
                    success = self._send_email_alert(message, alert_metadata)
                elif channel == "alertmanager" and self.alertmanager_url:
                    success = self._send_alertmanager_alert(message, alert_metadata)
                else:
                    self.logger.warning("Unsupported or unconfigured alert channel: %s", channel)
            except (requests.RequestException, ValueError, KeyError):
                self.logger.exception("Error sending %s alert", channel)
                success = False

        return success

    def _send_slack_alert(self, message: str, metadata: dict[str, Any]) -> bool:
        """Send an alert to Slack."""
        if not self.slack_webhook:
            return False

        # Format the message for Slack
        payload = {
            "text": f"*ALERT: {metadata['level']}*",
            "blocks": [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*ALERT: {metadata['level']}*\n{message}"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Context:*\n``````",
                    },
                },
            ],
        }

        response = requests.post(
            self.slack_webhook,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        success = 200 <= response.status_code < 300
        if not success:
            self.logger.error(
                "Slack alert failed with status %s: %s", response.status_code, response.text
            )

        return success

    def _send_email_alert(self, message: str, metadata: dict[str, Any]) -> bool:
        """Send an alert via email.

        Args:
            message: Alert message
            metadata: Alert metadata

        Returns:
            bool: Whether the email was sent successfully
        """
        # In a real implementation, you would use an email sending service
        # like SendGrid, AWS SES, or SMTP
        self.logger.info("Would send email to %s: %s", self.email_recipients, message)
        return True

    def _send_alertmanager_alert(self, message: str, metadata: dict[str, Any]) -> bool:
        """Send an alert to Alertmanager."""
        if not self.alertmanager_url:
            return False

        # Prepare Alertmanager alert payload
        payload = {
            "message": message,
            "description": json.dumps(metadata, indent=2),
            "priority": "P1" if metadata["level"] in ("CRITICAL", "ERROR") else "P2",
            "tags": ["dagster", "clustering", metadata["level"].lower()],
        }

        # Add optional fields
        if "run_id" in metadata:
            payload["alias"] = f"dagster-{metadata['run_id']}"

        response = requests.post(
            f"{self.alertmanager_url}/api/v1/alerts",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        success = 200 <= response.status_code < 300
        if not success:
            self.logger.error(
                "Alertmanager alert failed with status %s: %s", response.status_code, response.text
            )

        return success
