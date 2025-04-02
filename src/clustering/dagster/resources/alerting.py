"""Alerting service for Dagster pipelines."""

import json
import logging
import os
from typing import Dict, List, Optional, Union

import dagster as dg
import requests
from pydantic import BaseModel, EmailStr, Field


class AlertConfig(BaseModel):
    """Configuration for the alerting service."""

    enabled: bool = Field(True, description="Whether alerting is enabled")
    threshold: str = Field("ERROR", description="Minimum log level to trigger alerts")
    channels: List[str] = Field(["slack"], description="Alert channels to use")
    slack_webhook: Optional[str] = Field(None, description="Slack webhook URL")
    email_recipients: List[Union[str, EmailStr]] = Field([], description="Email recipients")
    opsgenie_api_key: Optional[str] = Field(None, description="OpsGenie API key")

    @property
    def threshold_level(self) -> int:
        """Get the numeric log level threshold."""
        return getattr(logging, self.threshold, logging.ERROR)


@dg.resource(config_schema=AlertConfig.model_json_schema())
def alerts_service(context: dg.InitResourceContext) -> "AlertingService":
    """Resource for sending alerts from Dagster pipelines.

    Args:
        context: The Dagster resource initialization context

    Returns:
        AlertingService: Service for sending alerts
    """
    config = context.resource_config

    # Get configuration with environment variable fallbacks
    slack_webhook = config.get("slack_webhook", os.environ.get("SLACK_WEBHOOK_URL"))
    opsgenie_api_key = config.get("opsgenie_api_key", os.environ.get("OPSGENIE_API_KEY"))

    return AlertingService(
        enabled=config.get("enabled", True),
        threshold=config.get("threshold", "ERROR"),
        channels=config.get("channels", ["slack"]),
        slack_webhook=slack_webhook,
        email_recipients=config.get("email_recipients", []),
        opsgenie_api_key=opsgenie_api_key,
        logger=context.log,
    )


class AlertingService:
    """Service for sending alerts from Dagster pipelines."""

    def __init__(
        self,
        enabled: bool,
        threshold: str,
        channels: List[str],
        slack_webhook: Optional[str],
        email_recipients: List[str],
        opsgenie_api_key: Optional[str],
        logger,
    ):
        self.enabled = enabled
        self.threshold = threshold
        self.threshold_level = getattr(logging, threshold, logging.ERROR)
        self.channels = channels
        self.slack_webhook = slack_webhook
        self.email_recipients = email_recipients
        self.opsgenie_api_key = opsgenie_api_key
        self.logger = logger

    def alert(
        self,
        message: str,
        level: str = "ERROR",
        context: Optional[Dict] = None,
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
            self.logger.info(f"Alerting disabled, would have sent: {message}")
            return False

        level_num = getattr(logging, level, logging.ERROR)
        if level_num < self.threshold_level:
            self.logger.debug(f"Alert level {level} below threshold {self.threshold}, not sending")
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
                elif channel == "opsgenie" and self.opsgenie_api_key:
                    success = self._send_opsgenie_alert(message, alert_metadata)
                else:
                    self.logger.warning(f"Unsupported or unconfigured alert channel: {channel}")
            except Exception as e:
                self.logger.error(f"Error sending {channel} alert: {e}")
                success = False

        return success

    def _send_slack_alert(self, message: str, metadata: Dict) -> bool:
        """Send an alert to Slack."""
        if not self.slack_webhook:
            return False

        # Format the message for Slack
        payload = {
            "text": f"*ALERT: {metadata['level']}*",
            "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": f"*ALERT: {metadata['level']}*\n{message}"}},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*Context:*\n```{json.dumps(metadata, indent=2)}```"},
                },
            ],
        }

        response = requests.post(self.slack_webhook, json=payload, headers={"Content-Type": "application/json"})

        success = 200 <= response.status_code < 300
        if not success:
            self.logger.error(f"Slack alert failed with status {response.status_code}: {response.text}")

        return success

    def _send_email_alert(self, message: str, metadata: Dict) -> bool:
        """Send an alert via email."""
        # In a real implementation, you would use an email sending service
        # like SendGrid, AWS SES, or SMTP
        self.logger.info(f"Would send email to {self.email_recipients}: {message}")
        return True

    def _send_opsgenie_alert(self, message: str, metadata: Dict) -> bool:
        """Send an alert to OpsGenie."""
        if not self.opsgenie_api_key:
            return False

        # Prepare OpsGenie alert payload
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
            "https://api.opsgenie.com/v2/alerts",
            json=payload,
            headers={"Content-Type": "application/json", "Authorization": f"GenieKey {self.opsgenie_api_key}"},
        )

        success = 200 <= response.status_code < 300
        if not success:
            self.logger.error(f"OpsGenie alert failed with status {response.status_code}: {response.text}")

        return success


@dg.failure_hook
def send_failure_alert(context):
    """Failure hook for sending alerts on pipeline failures.

    Args:
        context: The hook context
    """
    # Try to get the alerts service
    try:
        alerts = context.resources.alerts

        # Extract failure information
        failure_event = context.failure_event
        failure_metadata = failure_event.metadata
        step_key = context.step.key

        # Build meaningful error message
        error_message = f"Pipeline step '{step_key}' failed: {failure_event.message}"

        # Add context for the alert
        context_data = {
            "run_id": context.run_id,
            "step_key": step_key,
            "job_name": context.job_name,
            "failure_time": failure_event.timestamp,
        }

        if "error_stack" in failure_metadata:
            context_data["error_stack"] = failure_metadata["error_stack"]

        # Send the alert
        alerts.alert(
            message=error_message,
            level="ERROR",
            context=context_data,
        )
    except Exception as e:
        # Fallback logging if alerting fails
        context.log.error(f"Failed to send failure alert: {e}")
