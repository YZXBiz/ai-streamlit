"""Alerting resources for Dagster pipelines."""

import dagster as dg

from clustering.infra.monitoring import AlertingService


@dg.resource(
    config_schema={
        "enabled": dg.Field(dg.Bool, default_value=True, description="Whether alerting is enabled"),
        "threshold": dg.Field(dg.String, default_value="WARNING", description="Minimum log level to trigger alerts"),
        "slack_webhook": dg.Field(dg.Noneable(dg.String), default_value=None, description="Slack webhook URL"),
        "channels": dg.Field(dg.Array(dg.String), default_value=["slack"], description="Alert channels to use"),
        "email_recipients": dg.Field(dg.Array(dg.String), default_value=[], description="Email recipients"),
    }
)
def alerts_service(context: dg.InitResourceContext) -> AlertingService:
    """Resource for alerts.

    Args:
        context: The context for initializing the resource.

    Returns:
        AlertingService: A configured alerts service.
    """
    config = context.resource_config

    alerts = AlertingService(
        enabled=config.get("enabled", True),
        threshold=config.get("threshold", "WARNING"),
        channels=config.get("channels", ["slack"]),
        slack_webhook=config.get("slack_webhook"),
        email_recipients=config.get("email_recipients", []),
        alertmanager_url="http://localhost:9093",  # Default value
        logger=context.log,
    )
    alerts.start()

    @context.resource_cleanup_fn
    def cleanup_alerts():
        alerts.stop()

    return alerts
