"""Application settings configuration module."""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, Protocol, final


class Environment(str, Enum):
    """Valid environment types.

    String enum representing the different deployment environments
    """

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class LogLevel(str, Enum):
    """Valid logging levels.

    String enum representing standard logging levels
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SecretSettings:
    """Settings for secrets and credentials.

    Attributes:
        slack_webhook: Optional webhook URL for Slack notifications.
        api_keys: Dictionary of API keys indexed by service name.
    """

    slack_webhook: str | None = None
    api_keys: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        """Validate the settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Validate webhook URL format if present
        if self.slack_webhook and not self.slack_webhook.startswith(("http://", "https://")):
            errors.append(f"Invalid slack_webhook URL format: {self.slack_webhook}")

        return errors


@dataclass
class JobSettings:
    """Settings for job configurations.

    Attributes:
        name: The name of the job.
        enabled: Whether the job is enabled.
        schedule: Cron schedule for the job, if any.
        config_path: Path to the job configuration file.
        tags: Dictionary of tags to apply to the job.
    """

    name: str
    enabled: bool = True
    schedule: str | None = None
    config_path: str | None = None
    tags: dict[str, str] = field(default_factory=dict)

    # Class constants for use in job configurations
    DEFAULT_CONFIG_DIR: ClassVar[str] = "configs/job_configs"

    def validate(self) -> list[str]:
        """Validate the job settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Validate name isn't empty
        if not self.name:
            errors.append("Job name cannot be empty")

        # Validate config path exists if specified
        if self.config_path:
            config_file = Path(self.config_path)
            if not config_file.exists() and not config_file.is_absolute():
                # Try relative to project root if it's a relative path
                project_root = os.environ.get("PROJECT_ROOT", ".")
                alt_path = Path(project_root) / self.config_path
                if not alt_path.exists():
                    errors.append(f"Config file not found at: {self.config_path}")

        # Validate schedule format if specified (basic check)
        if self.schedule and len(self.schedule.split()) != 5:
            errors.append(f"Invalid cron schedule format: {self.schedule}")

        return errors


class Validatable(Protocol):
    """Protocol for objects that can be validated."""

    def validate(self) -> list[str]:
        """Validate the object.

        Returns:
            List of validation error messages (empty if valid)
        """
        ...


@final
@dataclass
class AppConfig:
    """Main application configuration.

    Attributes:
        env: Environment name (dev, staging, prod).
        log_level: Logging level.
        log_file: Path to log file.
        database_path: Path to DuckDB database file.
        database_schema: Database schema name.
        jobs: Dictionary of job settings indexed by job name.
        secrets: Secret configuration settings.
    """

    env: Environment = Environment.DEV
    log_level: LogLevel = LogLevel.INFO
    log_file: str = "logs.log"
    database_path: str = "outputs/clustering_dev.duckdb"
    database_schema: str = "public"
    jobs: dict[str, JobSettings] = field(default_factory=dict)
    secrets: SecretSettings = field(default_factory=SecretSettings)

    # Class constants
    VALID_SCHEMAS: ClassVar[list[str]] = ["public", "staging", "production"]

    def get_job_settings(self, job_name: str) -> JobSettings:
        """Get settings for a specific job.

        Args:
            job_name: Name of the job to retrieve settings for.

        Returns:
            JobSettings object for the specified job, or a default if not found.
        """
        return self.jobs.get(
            job_name,
            JobSettings(
                name=job_name,
                enabled=True,
                config_path=f"{JobSettings.DEFAULT_CONFIG_DIR}/{job_name}.yml",
            ),
        )

    def validate(self) -> list[str]:
        """Validate the configuration.

        Performs validation on all settings and sub-settings.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: list[str] = []

        # Validate database schema
        if self.database_schema not in self.VALID_SCHEMAS:
            errors.append(
                f"Invalid database schema: {self.database_schema}. "
                f"Valid options are: {', '.join(self.VALID_SCHEMAS)}"
            )

        # Validate all jobs
        for job_name, job_settings in self.jobs.items():
            job_errors = job_settings.validate()
            for error in job_errors:
                errors.append(f"Job '{job_name}': {error}")

        # Validate secrets
        secret_errors = self.secrets.validate()
        for error in secret_errors:
            errors.append(f"Secrets: {error}")

        return errors

    @classmethod
    def create_default(cls) -> "AppConfig":
        """Create a default configuration instance.

        Returns:
            Default AppConfig instance
        """
        return cls(
            env=Environment.DEV,
            log_level=LogLevel.INFO,
            log_file="logs.log",
            database_path="outputs/clustering_dev.duckdb",
            database_schema="public",
            jobs={
                "internal_preprocessing_job": JobSettings(
                    name="internal_preprocessing_job",
                    enabled=True,
                    config_path=f"{JobSettings.DEFAULT_CONFIG_DIR}/internal_preprocessing.yml",
                ),
                "internal_clustering_job": JobSettings(
                    name="internal_clustering_job",
                    enabled=True,
                    config_path=f"{JobSettings.DEFAULT_CONFIG_DIR}/internal_clustering.yml",
                ),
                "external_preprocessing_job": JobSettings(
                    name="external_preprocessing_job",
                    enabled=True,
                    config_path=f"{JobSettings.DEFAULT_CONFIG_DIR}/external_preprocessing.yml",
                ),
                "external_clustering_job": JobSettings(
                    name="external_clustering_job",
                    enabled=True,
                    config_path=f"{JobSettings.DEFAULT_CONFIG_DIR}/external_clustering.yml",
                ),
                "merging_job": JobSettings(
                    name="merging_job",
                    enabled=True,
                    config_path=f"{JobSettings.DEFAULT_CONFIG_DIR}/merging.yml",
                ),
                "full_pipeline_job": JobSettings(
                    name="full_pipeline_job",
                    enabled=True,
                    config_path=f"{JobSettings.DEFAULT_CONFIG_DIR}/full_pipeline.yml",
                ),
            },
            secrets=SecretSettings(),
        )


# Create a default configuration instance
CONFIG = AppConfig.create_default()
