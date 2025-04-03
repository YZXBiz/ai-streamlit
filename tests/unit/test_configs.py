"""Tests for the config module."""

from pathlib import Path

from clustering.io import config_parser


def test_parse_file(temp_config_file: Path) -> None:
    """Test parsing a YAML config file."""
    result = config_parser.parse_file(str(temp_config_file))

    # Check that the result is not empty
    assert result

    # Check the job kind is correct
    assert result["job"]["kind"] == "test_job"

    # Check parameters
    assert result["job"]["params"]["algorithm"] == "kmeans"
    assert result["job"]["params"]["n_clusters"] == 3


def test_parse_string() -> None:
    """Test parsing a config string."""
    config_str = """
    job:
      kind: string_test_job
      params:
        value: 42
    """

    result = config_parser.parse_string(config_str)

    assert result["job"]["kind"] == "string_test_job"
    assert result["job"]["params"]["value"] == 42


def test_merge_configs() -> None:
    """Test merging multiple configs."""
    config1 = {
        "job": {
            "kind": "test_job",
            "params": {
                "a": 1,
                "b": 2,
            },
        }
    }

    config2 = {
        "job": {
            "params": {
                "b": 3,  # Override
                "c": 4,  # New
            }
        }
    }

    merged = config_parser.merge_configs([config1, config2])

    assert merged["job"]["kind"] == "test_job"
    assert merged["job"]["params"]["a"] == 1
    assert merged["job"]["params"]["b"] == 3  # Overridden
    assert merged["job"]["params"]["c"] == 4  # Added


def test_to_object() -> None:
    """Test converting a config dict to a proper Python object."""
    config = {"job": {"kind": "test_job", "params": {"nested": {"value": 42}, "list": [1, 2, 3]}}}

    obj = config_parser.to_object(config)

    assert obj.job.kind == "test_job"
    assert obj.job.params.nested.value == 42
    assert obj.job.params.list == [1, 2, 3]
