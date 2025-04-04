"""Application settings configuration module."""

from dataclasses import dataclass, field


@dataclass
class SecretSettings:
    """Settings for secrets and credentials."""

    slack_webhook: str | None = None
    api_keys: dict[str, str] = field(default_factory=dict)


@dataclass
class JobSettings:
    """Settings for job configurations."""

    name: str
    enabled: bool = True
    schedule: str | None = None
    config_path: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Main application configuration."""

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
            job_name: Name of the job

        Returns:
            JobSettings for the specified job
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
