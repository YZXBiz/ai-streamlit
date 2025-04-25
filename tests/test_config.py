"""
Tests for the config module.

This module contains tests for the Config class in the config module.
"""

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from chatbot.config import Config


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Clean environment variables for testing."""
    # Backup and remove any config-related environment variables
    env_vars = [
        "OPENAI_API_KEY",
        "DB_PATH",
        "MEMORY_TYPE",
        "MEMORY_TOKEN_LIMIT",
        "LOG_LEVEL",
    ]

    old_values = {}
    for var in env_vars:
        old_values[var] = os.environ.get(var)
        monkeypatch.delenv(var, raising=False)

    yield

    # Restore environment variables after test
    for var, value in old_values.items():
        if value is not None:
            monkeypatch.setenv(var, value)


def test_config_defaults(clean_env: None) -> None:
    """Test default config values when no environment variables are set."""
    # Set minimum required env var
    os.environ["OPENAI_API_KEY"] = "dummy-api-key"

    config = Config()

    # Check default values
    assert config.DB_PATH == ":memory:"
    assert config.MEMORY_TYPE == "summary"
    assert config.MEMORY_TOKEN_LIMIT == 4000
    assert config.LOG_LEVEL == "INFO"
    assert config.OPENAI_API_KEY.get_secret_value() == "dummy-api-key"


def test_config_from_env_vars(clean_env: None) -> None:
    """Test loading config from environment variables."""
    # Set environment variables
    os.environ["OPENAI_API_KEY"] = "test-api-key"
    os.environ["DB_PATH"] = "test.duckdb"
    os.environ["MEMORY_TYPE"] = "simple"
    os.environ["MEMORY_TOKEN_LIMIT"] = "5000"
    os.environ["LOG_LEVEL"] = "DEBUG"

    config = Config()

    # Check values from environment
    assert config.OPENAI_API_KEY.get_secret_value() == "test-api-key"
    assert config.DB_PATH == "test.duckdb"
    assert config.MEMORY_TYPE == "simple"
    assert config.MEMORY_TOKEN_LIMIT == 5000
    assert config.LOG_LEVEL == "DEBUG"


def test_config_validation_api_key(clean_env: None) -> None:
    """Test validation of OPENAI_API_KEY."""
    # Test with empty API key
    with pytest.raises(ValidationError) as exc_info:
        os.environ["OPENAI_API_KEY"] = ""
        Config()

    # Verify the error message
    assert "OpenAI API key cannot be empty" in str(exc_info.value)


def test_config_validation_memory_type(clean_env: None) -> None:
    """Test validation of MEMORY_TYPE."""
    # Set valid API key
    os.environ["OPENAI_API_KEY"] = "dummy-api-key"

    # Test with invalid memory type
    with pytest.raises(ValidationError) as exc_info:
        os.environ["MEMORY_TYPE"] = "invalid"
        Config()

    # Verify the error message
    assert "Input should be 'simple' or 'summary'" in str(exc_info.value)


def test_config_validation_memory_token_limit(clean_env: None) -> None:
    """Test validation of MEMORY_TOKEN_LIMIT."""
    # Set valid API key
    os.environ["OPENAI_API_KEY"] = "dummy-api-key"

    # Test with token limit too low
    with pytest.raises(ValidationError) as exc_info:
        os.environ["MEMORY_TOKEN_LIMIT"] = "500"
        Config()

    # Verify the error message
    assert "Input should be greater than or equal to 1000" in str(exc_info.value)

    # Test with token limit too high
    with pytest.raises(ValidationError) as exc_info:
        os.environ["MEMORY_TOKEN_LIMIT"] = "20000"
        Config()

    # Verify the error message
    assert "Input should be less than or equal to 16000" in str(exc_info.value)


def test_config_validation_log_level(clean_env: None) -> None:
    """Test validation of LOG_LEVEL."""
    # Set valid API key
    os.environ["OPENAI_API_KEY"] = "dummy-api-key"

    # Test with invalid log level
    with pytest.raises(ValidationError) as exc_info:
        os.environ["LOG_LEVEL"] = "INVALID"
        Config()

    # Verify the error message
    assert "LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL" in str(exc_info.value)

    # Test with lowercase log level (should be normalized)
    os.environ["LOG_LEVEL"] = "debug"
    config = Config()
    assert config.LOG_LEVEL == "DEBUG"


@patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-valid-key"})
def test_config_instance() -> None:
    """Test that a config instance can be created with valid values."""
    config = Config()
    assert config.OPENAI_API_KEY.get_secret_value() == "sk-test-valid-key"
    assert config.DB_PATH == ":memory:"
    assert config.MEMORY_TYPE == "simple"
    assert config.MEMORY_TOKEN_LIMIT == 1000
    assert config.LOG_LEVEL == "DEBUG"
