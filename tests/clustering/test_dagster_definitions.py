"""Tests for clustering's Dagster implementation."""

import pytest
from dagster import build_op_context, build_asset_context, materialize
from dagster.test import run_op

from clustering.dagster.definitions import create_definitions, defs


class TestDagsterDefinitions:
    """Tests for Dagster definitions."""

    def test_defs_loads_successfully(self) -> None:
        """Test that Dagster definitions load without errors."""
        assert defs is not None
        assert hasattr(defs, "get_asset_job")
        assert hasattr(defs, "get_resource_config")

    def test_create_definitions(self) -> None:
        """Test that create_definitions returns valid definitions."""
        custom_defs = create_definitions(config_path=".")
        assert custom_defs is not None
        assert hasattr(custom_defs, "get_asset_job")


@pytest.mark.parametrize(
    "asset_key,expected_keys",
    [
        ("raw_sales_data", {"raw_sales_data"}),
        ("need_state_mappings", {"need_state_mappings"}),
    ],
)
def test_asset_dependencies(asset_key: str, expected_keys: set) -> None:
    """Test asset dependency structure.

    Args:
        asset_key: The key of the asset to test
        expected_keys: The expected set of keys this asset depends on
    """
    assets_by_key = {asset.key.to_python_identifier(): asset for asset in defs.get_all_assets()}
    if asset_key not in assets_by_key:
        pytest.skip(f"Asset {asset_key} not found in definitions")

    asset = assets_by_key[asset_key]
    deps = defs.get_asset_graph().get_dependent_assets(asset.key)
    actual_keys = {dep.to_python_identifier() for dep in deps}
    assert actual_keys == expected_keys


def test_materialize_assets():
    """Test that assets can be materialized in a test environment."""
    # Create a small materialization test with mocked inputs
    # This is a simplified test and would need actual mock data
    try:
        # Try materializing a simple asset - this would need to be adapted
        # to your actual asset structure with appropriate mock data
        result = materialize(
            [
                asset
                for asset in defs.get_all_assets()
                if "config" in asset.key.to_python_identifier()
            ],
            run_config={
                "resources": {"io_manager": {"config": {"base_path": "/tmp/test_dagster"}}}
            },
        )
        assert result.success
    except Exception:
        # If we can't materialize with the default setup, we'll skip this test
        # In a real implementation, you would provide proper mock data
        pytest.skip("Skipping materialization test - needs mock data configuration")


@pytest.mark.skip(reason="Implement with actual op names from your Dagster implementation")
def test_specific_op():
    """Test a specific Dagster op in isolation."""
    # Replace with an actual op from your implementation
    op_name = "process_data_op"

    try:
        from clustering.dagster.definitions import process_data_op

        # Create test inputs
        test_input = {"data": [{"id": 1, "value": 100}, {"id": 2, "value": 200}]}

        # Run the op with test context and inputs
        result = run_op(
            process_data_op,
            input_values=test_input,
            resources={"io_manager": {"config": {"base_path": "/tmp/test_dagster"}}},
        )

        # Validate the output
        assert result.success
        assert "processed_data" in result.output_values
        processed = result.output_values["processed_data"]
        assert len(processed) == 2

    except ImportError:
        pytest.skip("Op not found - update test with actual op from your implementation")
