"""Tests for Dagster resources in clustering pipeline."""

import os
import tempfile
from pathlib import Path

import pytest
from dagster import build_init_resource_context, build_resources

from clustering.dagster.definitions import defs


class TestDagsterResources:
    """Tests for Dagster resources."""

    def test_resources_defined(self) -> None:
        """Test that expected resources are defined in the definitions."""
        resource_keys = list(defs.get_resource_defs().keys())
        assert len(resource_keys) > 0

        # Common resources to expect - modify based on your actual implementation
        expected_resources = ["io_manager", "config", "logger"]

        for resource in expected_resources:
            assert any(resource in key for key in resource_keys), (
                f"Expected resource {resource} not found"
            )

    @pytest.mark.parametrize(
        "resource_key",
        [
            "io_manager",
            "config",  # Assume you have a config resource
            # Add other resources to test
        ],
    )
    def test_resource_initialization(self, resource_key: str) -> None:
        """Test that resources can be initialized.

        Args:
            resource_key: The key of the resource to test
        """
        # Skip if resource doesn't exist
        if resource_key not in defs.get_resource_defs():
            pytest.skip(f"Resource {resource_key} not found in definitions")

        try:
            # Create a temporary directory for any file resources
            with tempfile.TemporaryDirectory() as temp_dir:
                # Initialize resources with minimal config
                resource_defs = {
                    k: v
                    for k, v in defs.get_resource_defs().items()
                    if k == resource_key or k.startswith("__")  # Include target and builtins
                }

                # Customize config based on resource type
                resource_config = {}
                if resource_key == "io_manager":
                    resource_config = {"config": {"base_path": temp_dir}}
                elif resource_key == "config":
                    # Create a minimal config file
                    config_path = Path(temp_dir) / "config.yml"
                    with open(config_path, "w") as f:
                        f.write("job:\n  kind: test\n  params:\n    test: value\n")
                    resource_config = {"config_path": str(config_path)}

                # Initialize the resource
                context = build_init_resource_context(config=resource_config)
                resource_def = defs.get_resource_defs()[resource_key]
                resource = resource_def.instantiate_at_context(context)

                # Basic assertion - resource should be created without errors
                assert resource is not None

        except Exception as e:
            pytest.fail(f"Failed to initialize resource {resource_key}: {str(e)}")

    def test_resource_with_context(self) -> None:
        """Test resources using the build_resources context manager."""
        try:
            # Only select a few resources to test - replace with actual keys from your config
            resource_keys_to_test = ["io_manager"]

            # Filter to only get resources we want to test
            resource_defs = {
                k: v
                for k, v in defs.get_resource_defs().items()
                if k in resource_keys_to_test or k.startswith("__")  # Include builtins
            }

            # Initialize resources
            with tempfile.TemporaryDirectory() as temp_dir:
                with build_resources(
                    resource_defs=resource_defs,
                    resource_config={"io_manager": {"config": {"base_path": temp_dir}}},
                ) as resources:
                    # Test that resources are initialized correctly
                    assert resources.io_manager is not None

                    # If you have specific methods to test, do that here
                    if hasattr(resources.io_manager, "get_base_path"):
                        path = resources.io_manager.get_base_path()
                        assert path == temp_dir
        except Exception as e:
            pytest.fail(f"Failed to test resources with context: {str(e)}")

    @pytest.mark.skip(reason="Customize for specific resource integration tests")
    def test_resource_integration(self) -> None:
        """Test that resources work together correctly."""
        # This would be a more complex test that shows how different resources
        # integrate with each other in your specific implementation

        # Example pattern (requires adaptation to your specific resources):
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                with build_resources(
                    resource_defs=defs.get_resource_defs(),
                    resource_config={
                        "io_manager": {"config": {"base_path": temp_dir}},
                        # Add config for other resources
                    },
                ) as resources:
                    # Test interaction between resources
                    # For example, if you have a resource that depends on io_manager
                    if hasattr(resources, "data_loader") and hasattr(resources, "io_manager"):
                        # Test that data_loader can use io_manager
                        pass
        except Exception:
            pytest.skip("Resource integration test needs to be customized")


# Add more specific tests for each resource type based on your implementation
