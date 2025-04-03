"""Tests for IO configs module."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from clustering.io.config_parser import load_config, save_config


def test_load_config_json():
    """Test loading a config from a JSON file."""
    with TemporaryDirectory() as temp_dir:
        # Create a sample config file
        config_path = Path(temp_dir) / "test_config.json"
        config_data = {
            "name": "test",
            "version": 1,
            "parameters": {"param1": 100, "param2": "value", "nested": {"key": "value"}},
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f)

        # Load the config
        loaded_config = load_config(config_path)

        # Check that the config was loaded correctly
        assert loaded_config == config_data


def test_load_config_yaml():
    """Test loading a config from a YAML file."""
    pytest.importorskip("yaml")  # Skip test if yaml is not installed

    with TemporaryDirectory() as temp_dir:
        # Create a sample config file
        config_path = Path(temp_dir) / "test_config.yaml"
        config_str = """
        name: test
        version: 1
        parameters:
          param1: 100
          param2: value
          nested:
            key: value
        """

        with open(config_path, "w") as f:
            f.write(config_str)

        # Load the config
        loaded_config = load_config(config_path)

        # Check that the config was loaded correctly
        expected = {
            "name": "test",
            "version": 1,
            "parameters": {"param1": 100, "param2": "value", "nested": {"key": "value"}},
        }
        assert loaded_config == expected


def test_load_config_unsupported_format():
    """Test loading a config with an unsupported file format."""
    with TemporaryDirectory() as temp_dir:
        # Create a file with an unsupported extension
        config_path = Path(temp_dir) / "test_config.txt"
        with open(config_path, "w") as f:
            f.write("This is not a supported config format")

        # Loading an unsupported format should raise a ValueError
        with pytest.raises(ValueError):
            load_config(config_path)


def test_save_config_json():
    """Test saving a config to a JSON file."""
    with TemporaryDirectory() as temp_dir:
        # Define a config to save
        config_path = Path(temp_dir) / "test_config.json"
        config_data = {
            "name": "test",
            "version": 1,
            "parameters": {"param1": 100, "param2": "value", "nested": {"key": "value"}},
        }

        # Save the config
        save_config(config_data, config_path)

        # Check that the file was created
        assert os.path.exists(config_path)

        # Load the config back and verify contents
        with open(config_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data == config_data


def test_save_config_yaml():
    """Test saving a config to a YAML file."""
    pytest.importorskip("yaml")  # Skip test if yaml is not installed

    with TemporaryDirectory() as temp_dir:
        # Define a config to save
        config_path = Path(temp_dir) / "test_config.yaml"
        config_data = {
            "name": "test",
            "version": 1,
            "parameters": {"param1": 100, "param2": "value", "nested": {"key": "value"}},
        }

        # Save the config
        save_config(config_data, config_path)

        # Check that the file was created
        assert os.path.exists(config_path)

        # Load the config back and verify contents
        loaded_config = load_config(config_path)
        assert loaded_config == config_data


def test_save_config_unsupported_format():
    """Test saving a config with an unsupported file format."""
    with TemporaryDirectory() as temp_dir:
        # Define a config to save
        config_path = Path(temp_dir) / "test_config.txt"
        config_data = {"key": "value"}

        # Saving to an unsupported format should raise a ValueError
        with pytest.raises(ValueError):
            save_config(config_data, config_path)


def test_save_config_parent_dirs():
    """Test that save_config creates parent directories."""
    with TemporaryDirectory() as temp_dir:
        # Define a config path with nested directories
        nested_path = Path(temp_dir) / "nested" / "dirs" / "test_config.json"
        config_data = {"key": "value"}

        # Save the config
        save_config(config_data, nested_path)

        # Check that the parent directories and file were created
        assert os.path.exists(nested_path)


@pytest.fixture
def temp_config_file() -> Path:
    """Create a temporary config file for testing."""
    with TemporaryDirectory() as temp_dir:
        # Create a sample config file
        config_path = Path(temp_dir) / "test_config.yml"
        config_data = {"job": {"kind": "test_job", "params": {"value": 42}}}

        with open(config_path, "w") as f:
            json.dump(config_data, f)

    yield config_path

    # Clean up
    if config_path.exists():
        config_path.unlink()


def test_load_config(temp_config_file: Path) -> None:
    """Test loading a config file."""
    config = load_config(str(temp_config_file))

    assert config["job"]["kind"] == "test_job"
    assert config["job"]["params"]["value"] == 42


def test_save_config(temp_config_file: Path) -> None:
    """Test saving a config file."""
    # Create a test config
    config = {"job": {"kind": "save_test_job", "params": {"value": 100}}}

    # Save to a new path
    new_path = str(temp_config_file.with_suffix(".new.yml"))
    save_config(config, new_path)

    # Check that the file was created
    assert os.path.exists(new_path)

    # Load and verify
    loaded_config = load_config(new_path)
    assert loaded_config["job"]["kind"] == "save_test_job"
    assert loaded_config["job"]["params"]["value"] == 100

    # Clean up
    os.unlink(new_path)


def test_save_config_nested_path() -> None:
    """Test saving a config file to a nested path that doesn't exist."""
    # Create a nested path
    with TemporaryDirectory() as temp_dir:
        nested_dir = os.path.join(temp_dir, "nested", "dirs")
        nested_path = os.path.join(nested_dir, "config.yml")

        # Create a test config
        config = {"job": {"kind": "nested_test_job"}}

        # Save to the nested path
        save_config(config, nested_path)

        # Check that the parent directories and file were created
        assert os.path.exists(nested_path)
