"""Application settings configuration module."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SecretSettings:
    """Settings for secrets and credentials.
    
    Attributes:
        slack_webhook: Optional webhook URL for Slack notifications.
        api_keys: Dictionary of API keys indexed by service name.
    """

    slack_webhook: str | None = None
    api_keys: dict[str, str] = field(default_factory=dict)


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

    env: str = "dev"
    log_level: str = "INFO"
    log_file: str = "logs.log"
    database_path: str = "outputs/clustering_dev.duckdb"
    database_schema: str = "public"
    jobs: dict[str, JobSettings] = field(default_factory=dict)
    secrets: SecretSettings = field(default_factory=SecretSettings)

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
                config_path=f"configs/job_configs/{job_name}.yml",
            ),
        )


# Create a default configuration instance
CONFIG = AppConfig(
    env="dev",
    log_level="INFO",
    log_file="logs.log",
    database_path="outputs/clustering_dev.duckdb",
    database_schema="public",
    jobs={
        "internal_preprocessing_job": JobSettings(
            name="internal_preprocessing_job",
            enabled=True,
            config_path="configs/job_configs/internal_preprocessing.yml",
        ),
        "internal_clustering_job": JobSettings(
            name="internal_clustering_job",
            enabled=True,
            config_path="configs/job_configs/internal_clustering.yml",
        ),
        "external_preprocessing_job": JobSettings(
            name="external_preprocessing_job",
            enabled=True,
            config_path="configs/job_configs/external_preprocessing.yml",
        ),
        "external_clustering_job": JobSettings(
            name="external_clustering_job",
            enabled=True, 
            config_path="configs/job_configs/external_clustering.yml",
        ),
        "merging_job": JobSettings(
            name="merging_job",
            enabled=True,
            config_path="configs/job_configs/merging.yml",
        ),
        "full_pipeline_job": JobSettings(
            name="full_pipeline_job",
            enabled=True,
            config_path="configs/job_configs/full_pipeline.yml",
        ),
    },
    secrets=SecretSettings(),
)
