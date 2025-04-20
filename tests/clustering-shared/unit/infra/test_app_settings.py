"""Tests for the application settings module."""

import os
from pathlib import Path
import pytest
from unittest.mock import patch

from clustering.shared.infra.app_settings import (
    Environment,
    LogLevel,
    SecretSettings,
    JobSettings,
    AppConfig,
    CONFIG,
)


class TestEnvironment:
    """Tests for the Environment enum."""

    def test_environment_values(self) -> None:
        """Test that the Environment enum has the correct values."""
        assert Environment.DEV == "dev"
        assert Environment.STAGING == "staging"
        assert Environment.PROD == "prod"


class TestLogLevel:
    """Tests for the LogLevel enum."""

    def test_log_level_values(self) -> None:
        """Test that the LogLevel enum has the correct values."""
        assert LogLevel.DEBUG == "DEBUG"
        assert LogLevel.INFO == "INFO"
        assert LogLevel.WARNING == "WARNING"
        assert LogLevel.ERROR == "ERROR"
        assert LogLevel.CRITICAL == "CRITICAL"


class TestSecretSettings:
    """Tests for the SecretSettings class."""

    def test_default_initialization(self) -> None:
        """Test that SecretSettings initializes with correct defaults."""
        settings = SecretSettings()
        assert settings.slack_webhook is None
        assert settings.api_keys == {}

    def test_custom_initialization(self) -> None:
        """Test that SecretSettings initializes with custom values."""
        settings = SecretSettings(
            slack_webhook="https://hooks.slack.com/services/123456",
            api_keys={"service1": "key1", "service2": "key2"},
        )
        assert settings.slack_webhook == "https://hooks.slack.com/services/123456"
        assert settings.api_keys == {"service1": "key1", "service2": "key2"}

    def test_validate_valid_settings(self) -> None:
        """Test validation of valid settings."""
        settings = SecretSettings(slack_webhook="https://hooks.slack.com/services/123456")
        errors = settings.validate()
        assert errors == []

    def test_validate_invalid_webhook(self) -> None:
        """Test validation of invalid webhook URL."""
        settings = SecretSettings(slack_webhook="invalid-url")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Invalid slack_webhook URL format" in errors[0]


class TestJobSettings:
    """Tests for the JobSettings class."""

    def test_default_initialization(self) -> None:
        """Test that JobSettings initializes with correct defaults."""
        settings = JobSettings(name="test_job")
        assert settings.name == "test_job"
        assert settings.enabled is True
        assert settings.schedule is None
        assert settings.config_path is None
        assert settings.tags == {}

    def test_custom_initialization(self) -> None:
        """Test that JobSettings initializes with custom values."""
        settings = JobSettings(
            name="test_job",
            enabled=False,
            schedule="0 0 * * *",
            config_path="configs/test_job.yml",
            tags={"owner": "test_team"},
        )
        assert settings.name == "test_job"
        assert settings.enabled is False
        assert settings.schedule == "0 0 * * *"
        assert settings.config_path == "configs/test_job.yml"
        assert settings.tags == {"owner": "test_team"}

    def test_validate_valid_settings(self) -> None:
        """Test validation of valid settings."""
        settings = JobSettings(name="test_job")
        errors = settings.validate()
        assert errors == []

    def test_validate_empty_name(self) -> None:
        """Test validation of empty job name."""
        settings = JobSettings(name="")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Job name cannot be empty" in errors[0]

    @patch("pathlib.Path.exists")
    def test_validate_nonexistent_config_path(self, mock_exists) -> None:
        """Test validation of nonexistent config path."""
        mock_exists.return_value = False
        settings = JobSettings(name="test_job", config_path="nonexistent.yml")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Config file not found" in errors[0]

    def test_validate_invalid_schedule(self) -> None:
        """Test validation of invalid schedule format."""
        settings = JobSettings(name="test_job", schedule="invalid")
        errors = settings.validate()
        assert len(errors) == 1
        assert "Invalid cron schedule format" in errors[0]


class TestAppConfig:
    """Tests for the AppConfig class."""

    def test_default_initialization(self) -> None:
        """Test that AppConfig initializes with correct defaults."""
        config = AppConfig()
        assert config.env == Environment.DEV
        assert config.log_level == LogLevel.INFO
        assert config.log_file == "logs.log"
        assert config.database_path == "outputs/clustering_dev.duckdb"
        assert config.database_schema == "public"
        assert isinstance(config.jobs, dict)
        assert isinstance(config.secrets, SecretSettings)

    def test_get_job_settings_existing(self) -> None:
        """Test retrieving existing job settings."""
        config = AppConfig()
        job_settings = JobSettings(name="test_job", enabled=False)
        config.jobs["test_job"] = job_settings
        
        result = config.get_job_settings("test_job")
        assert result is job_settings
        assert result.enabled is False

    def test_get_job_settings_nonexistent(self) -> None:
        """Test retrieving non-existent job settings."""
        config = AppConfig()
        result = config.get_job_settings("nonexistent_job")
        assert result.name == "nonexistent_job"
        assert result.enabled is True
        assert result.config_path == f"{JobSettings.DEFAULT_CONFIG_DIR}/nonexistent_job.yml"

    def test_validate_valid_config(self) -> None:
        """Test validation of valid config."""
        config = AppConfig()
        errors = config.validate()
        assert errors == []

    def test_validate_invalid_schema(self) -> None:
        """Test validation of invalid database schema."""
        config = AppConfig(database_schema="invalid_schema")
        errors = config.validate()
        assert len(errors) == 1
        assert "Invalid database schema" in errors[0]

    def test_validate_propagates_job_errors(self) -> None:
        """Test that validation propagates job errors."""
        config = AppConfig()
        config.jobs["test_job"] = JobSettings(name="")
        errors = config.validate()
        assert len(errors) == 1
        assert "Job 'test_job': Job name cannot be empty" in errors[0]

    def test_validate_propagates_secret_errors(self) -> None:
        """Test that validation propagates secret errors."""
        config = AppConfig()
        config.secrets = SecretSettings(slack_webhook="invalid-url")
        errors = config.validate()
        assert len(errors) == 1
        assert "Secrets: Invalid slack_webhook URL format" in errors[0]

    def test_create_default(self) -> None:
        """Test creating default configuration."""
        config = AppConfig.create_default()
        assert config.env == Environment.DEV
        assert config.log_level == LogLevel.INFO
        assert len(config.jobs) == 6  # Verify all default jobs are created
        assert "internal_preprocessing_job" in config.jobs
        assert "internal_clustering_job" in config.jobs
        assert "external_preprocessing_job" in config.jobs
        assert "external_clustering_job" in config.jobs
        assert "merging_job" in config.jobs
        assert "full_pipeline_job" in config.jobs

    def test_global_config_instance(self) -> None:
        """Test the global CONFIG instance."""
        assert CONFIG is not None
        assert isinstance(CONFIG, AppConfig)
        assert CONFIG.env == Environment.DEV 