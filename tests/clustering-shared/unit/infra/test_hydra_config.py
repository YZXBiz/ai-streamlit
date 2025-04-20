"""Tests for the hydra configuration module."""

import os
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, mock_open

from clustering.shared.infra.hydra_config import (
    parse_file,
    parse_string,
    merge_configs,
    to_object,
    load_config,
)


@pytest.fixture
def config_file() -> Path:
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
        temp_path = Path(temp.name)
        temp.write(
            b"""
            app:
              name: TestApp
              version: 1.0.0
            database:
              host: localhost
              port: 5432
            """
        )
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        os.unlink(temp_path)


class TestHydraConfig:
    """Tests for hydra configuration functions."""

    def test_parse_file_existing(self, config_file: Path) -> None:
        """Test parsing an existing config file."""
        config = parse_file(str(config_file))
        assert config is not None
        assert "app" in config
        assert config.app.name == "TestApp"
        assert config.app.version == "1.0.0"
        assert "database" in config
        assert config.database.host == "localhost"
        assert config.database.port == 5432

    def test_parse_file_nonexistent(self) -> None:
        """Test parsing a nonexistent config file."""
        config = parse_file("/nonexistent/path/config.yaml")
        assert config is None

    def test_parse_string(self) -> None:
        """Test parsing a config string."""
        yaml_str = """
        app:
          name: TestApp
          version: 1.0.0
        """
        config = parse_string(yaml_str)
        assert config is not None
        assert "app" in config
        assert config.app.name == "TestApp"
        assert config.app.version == "1.0.0"

    def test_merge_configs(self) -> None:
        """Test merging multiple configs."""
        config1 = parse_string("""
        app:
          name: TestApp
          version: 1.0.0
        """)
        
        config2 = parse_string("""
        database:
          host: localhost
          port: 5432
        """)
        
        merged = merge_configs([config1, config2])
        assert merged is not None
        assert "app" in merged
        assert merged.app.name == "TestApp"
        assert merged.app.version == "1.0.0"
        assert "database" in merged
        assert merged.database.host == "localhost"
        assert merged.database.port == 5432

    def test_to_object(self) -> None:
        """Test converting a config to a Python object."""
        config = parse_string("""
        app:
          name: TestApp
          version: 1.0.0
        list_test:
          - item1
          - item2
        """)
        
        obj = to_object(config)
        assert isinstance(obj, dict)
        assert "app" in obj
        assert obj["app"]["name"] == "TestApp"
        assert obj["app"]["version"] == "1.0.0"
        assert "list_test" in obj
        assert isinstance(obj["list_test"], list)
        assert obj["list_test"] == ["item1", "item2"]

    def test_to_object_with_variables(self) -> None:
        """Test converting a config with variables to a Python object."""
        # Use a simpler string without unregistered ENV variables
        config = parse_string("""
        app:
          name: TestApp
          version: "1.0.0"
        """)
        
        # Test simple conversion
        obj = to_object(config)
        assert obj["app"]["name"] == "TestApp"
        assert obj["app"]["version"] == "1.0.0"
        
        # For resolve=False, we should test with a number that would be resolved
        config2 = parse_string("""
        numbers:
          a: 10
          b: 20
          c: ${numbers.a}
        """)
        
        # With resolve=True, c should be 10
        obj_resolved = to_object(config2)
        assert obj_resolved["numbers"]["c"] == 10
        
        # With resolve=False, c should remain as the interpolation string
        obj_unresolved = to_object(config2, resolve=False)
        assert obj_unresolved["numbers"]["c"] == "${numbers.a}"

    def test_load_config(self, config_file: Path) -> None:
        """Test loading a config file and converting to Python object."""
        config = load_config(str(config_file))
        assert config is not None
        assert isinstance(config, dict)
        assert "app" in config
        assert config["app"]["name"] == "TestApp"
        assert config["app"]["version"] == "1.0.0"
        assert "database" in config
        assert config["database"]["host"] == "localhost"
        assert config["database"]["port"] == 5432

    def test_load_config_nonexistent(self) -> None:
        """Test loading a nonexistent config file."""
        config = load_config("/nonexistent/path/config.yaml")
        assert config is None 