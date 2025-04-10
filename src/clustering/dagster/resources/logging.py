"""Logging resources for Dagster pipelines."""

import logging
from typing import Any, Optional

import dagster as dg


class LoggerService:
    """Logger service for Dagster pipelines."""

    def __init__(self, log_level: str = "INFO"):
        """Initialize logger service.

        Args:
            log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_level = getattr(logging, log_level)
        self.logger = logging.getLogger("clustering_pipeline")
        self.logger.setLevel(self.log_level)

    def log(self, level: str, message: str):
        """Log a message at the specified level.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
        """
        log_method = getattr(self.logger, level.lower())
        log_method(message)


# Define logger service as a resource
@dg.resource
def logger_service(context):
    """Create a logger service.

    Args:
        context: Resource initialization context

    Returns:
        Logger service instance
    """
    return LoggerService(log_level=context.resource_config.get("log_level", "INFO"))


class AlertsService:
    """Alert service for Dagster pipelines."""
    
    def __init__(self, enabled: bool = True, threshold: str = "WARNING", slack_webhook: Optional[str] = None):
        """Initialize alerts service.
        
        Args:
            enabled: Whether alerts are enabled
            threshold: Minimum severity level to trigger alerts (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            slack_webhook: Optional Slack webhook URL for sending alerts
        """
        self.enabled = enabled
        self.threshold = getattr(logging, threshold)
        self.slack_webhook = slack_webhook
        self.logger = logging.getLogger("clustering_alerts")
        
    def send_alert(self, level: str, message: str):
        """Send an alert if the level meets the threshold.
        
        Args:
            level: Alert level (debug, info, warning, error, critical)
            message: Alert message
        """
        if not self.enabled:
            return
            
        level_num = getattr(logging, level.upper())
        if level_num < self.threshold:
            return
            
        # Log the alert
        self.logger.log(level_num, f"ALERT: {message}")
        
        # Send to Slack if configured
        if self.slack_webhook and level_num >= logging.ERROR:
            # In a real implementation, this would send to Slack
            self.logger.info(f"Would send to Slack: {message}")


# Define alerts service as a resource
@dg.resource
def alerts_service(context):
    """Create an alerts service.
    
    Args:
        context: Resource initialization context
        
    Returns:
        Alerts service instance
    """
    return AlertsService(
        enabled=context.resource_config.get("enabled", True),
        threshold=context.resource_config.get("threshold", "WARNING"),
        slack_webhook=context.resource_config.get("slack_webhook"),
    )
