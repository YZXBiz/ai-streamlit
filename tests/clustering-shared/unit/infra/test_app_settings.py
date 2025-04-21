"""Tests for the application settings module."""

import os
from unittest import mock

import pytest

from clustering.shared.infra.app_settings import (
    AppConfig,
    CONFIG,
    Environment,
    JobSettings,
    LogLevel,
    SecretSettings,
)


class TestEnvironment:
    """Tests for Environment enum."""

    def test_environment_values(self):
        """Test Environment enum has the expected values."""
        assert Environment.DEV == "dev"
        assert Environment.STAGING == "staging"
        assert Environment.PROD == "prod"


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_log_level_values(self):
        """Test LogLevel enum has the expected values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"


class TestSecretSettings:
    """Tests for SecretSettings class."""

    def test_default_constructor(self):
        """Test SecretSettings default constructor."""
        settings = SecretSettings()
        assert settings.slack_webhook is None
        assert settings.api_keys == {}

    def test_custom_constructor(self):
        """Test SecretSettings custom constructor."""
        settings = SecretSettings(
            slack_webhook="https://hooks.slack.com/services/123",
            api_keys={"api1": "key1", "api2": "key2"},
        )
        assert settings.slack_webhook == "https://hooks.slack.com/services/123"
        assert settings.api_keys == {"api1": "key1", "api2": "key2"}

    def test_validate_valid_settings(self):
        """Test validation of valid settings."""
        settings = SecretSettings(slack_webhook="https://hooks.slack.com/services/123")
        errors = settings.validate()
        assert errors == []

    def test_validate_invalid_webhook(self):
        """Test validation of invalid webhook."""
        settings = SecretSettings(slack_webhook="invalid-url")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Invalid slack_webhook URL format" in errors[0]


class TestJobSettings:
    """Tests for JobSettings class."""

    def test_default_constructor(self):
        """Test JobSettings default constructor."""
        settings = JobSettings(name="test_job")
        assert settings.name == "test_job"
        assert settings.enabled is True
        assert settings.schedule is None
        assert settings.config_path is None
        assert settings.tags == {}

    def test_custom_constructor(self):
        """Test JobSettings custom constructor."""
        settings = JobSettings(
            name="test_job",
            enabled=False,
            schedule="0 * * * *",
            config_path="configs/test.yml",
            tags={"key": "value"},
        )
        assert settings.name == "test_job"
        assert settings.enabled is False
        assert settings.schedule == "0 * * * *"
        assert settings.config_path == "configs/test.yml"
        assert settings.tags == {"key": "value"}

    def test_validate_valid_settings(self):
        """Test validation of valid settings."""
        settings = JobSettings(name="test_job", schedule="* * * * *")
        errors = settings.validate()
        assert errors == []

    def test_validate_empty_name(self):
        """Test validation of empty name."""
        settings = JobSettings(name="")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Job name cannot be empty" in errors[0]

    def test_validate_invalid_schedule(self):
        """Test validation of invalid schedule."""
        settings = JobSettings(name="test_job", schedule="invalid")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Invalid cron schedule format" in errors[0]

    @mock.patch("pathlib.Path.exists")
    def test_validate_config_file_not_found(self, mock_exists):
        """Test validation of config file not found."""
        mock_exists.return_value = False
        settings = JobSettings(name="test_job", config_path="configs/missing.yml")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Config file not found" in errors[0]

    @mock.patch("pathlib.Path.exists")
    def test_validate_config_file_found(self, mock_exists):
        """Test validation of config file found."""
        mock_exists.return_value = True
        settings = JobSettings(name="test_job", config_path="configs/existing.yml")
        errors = settings.validate()
        assert errors == []


class TestAppConfig:
    """Tests for AppConfig class."""

    def test_default_constructor(self):
        """Test AppConfig default constructor."""
        config = AppConfig()
        assert config.env == Environment.DEV
        assert config.log_level == LogLevel.INFO
        assert config.log_file == "logs.log"
        assert config.database_path == "outputs/clustering_dev.duckdb"
        assert config.database_schema == "public"
        assert config.jobs == {}
        assert isinstance(config.secrets, SecretSettings)

    def test_custom_constructor(self):
        """Test AppConfig custom constructor."""
        jobs = {
            "job1": JobSettings(name="job1"),
            "job2": JobSettings(name="job2"),
        }
        secrets = SecretSettings(slack_webhook="https://hooks.slack.com/services/123")

        config = AppConfig(
            env=Environment.PROD,
            log_level=LogLevel.ERROR,
            log_file="prod.log",
            database_path="outputs/prod.duckdb",
            database_schema="production",
            jobs=jobs,
            secrets=secrets,
        )

        assert config.env == Environment.PROD
        assert config.log_level == LogLevel.ERROR
        assert config.log_file == "prod.log"
        assert config.database_path == "outputs/prod.duckdb"
        assert config.database_schema == "production"
        assert config.jobs == jobs
        assert config.secrets == secrets

    def test_get_job_settings_existing(self):
        """Test get_job_settings for an existing job."""
        job_settings = JobSettings(name="test_job")
        config = AppConfig(jobs={"test_job": job_settings})
        result = config.get_job_settings("test_job")
        assert result == job_settings

    def test_get_job_settings_nonexistent(self):
        """Test get_job_settings for a nonexistent job."""
        config = AppConfig()
        result = config.get_job_settings("nonexistent_job")
        assert result.name == "nonexistent_job"
        assert result.enabled is True
        assert result.config_path == "configs/job_configs/nonexistent_job.yml"

    def test_validate_valid_schema(self):
        """Test validation of valid schema."""
        config = AppConfig(database_schema="public")
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_schema(self):
        """Test validation of invalid schema."""
        config = AppConfig(database_schema="invalid")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid database schema" in errors[0]

    def test_validate_with_job_errors(self):
        """Test validation with job errors."""
        job = JobSettings(name="", schedule="invalid")  # Two errors
        config = AppConfig(jobs={"test_job": job})
        errors = config.validate()
        assert len(errors) == 2
        assert all("Job 'test_job'" in error for error in errors)

    def test_validate_with_secret_errors(self):
        """Test validation with secret errors."""
        secret = SecretSettings(slack_webhook="invalid-url")
        config = AppConfig(secrets=secret)
        errors = config.validate()
        assert len(errors) == 1
        assert "Secrets:" in errors[0]

    def test_create_default(self):
        """Test create_default method."""
        config = AppConfig.create_default()
        assert config.env == Environment.DEV
        assert config.log_level == LogLevel.INFO
        assert len(config.jobs) == 6
        assert "internal_preprocessing_job" in config.jobs
        assert "full_pipeline_job" in config.jobs


def test_config_is_instance():
    """Test CONFIG is an instance of AppConfig."""
    assert isinstance(CONFIG, AppConfig)
