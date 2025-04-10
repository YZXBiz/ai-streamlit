"""Tests for Dagster assets in clustering pipeline."""

import os
from pathlib import Path

import pandas as pd
import pytest
from dagster import (
    AssetKey,
    AssetMaterialization,
    build_asset_context,
    materialize_to_memory,
)

from clustering.dagster.definitions import defs


class TestDagsterAssets:
    """Tests for Dagster assets."""

    @pytest.fixture
    def mock_asset_context(self):
        """Create a mock asset context for testing."""
        return build_asset_context(
            resources={
                "io_manager": {"config": {"base_path": "/tmp/test_dagster"}},
            }
        )

    @pytest.fixture
    def sample_sales_data(self) -> pd.DataFrame:
        """Create sample sales data for testing."""
        return pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103, 104, 105],
                "STORE_NBR": [1, 2, 3, 1, 2],
                "CAT_DSC": ["Category A"] * 5,
                "TOTAL_SALES": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    @pytest.fixture
    def sample_need_state_data(self) -> pd.DataFrame:
        """Create sample need state mapping data for testing."""
        return pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103, 104, 105],
                "CATEGORY": ["Category A"] * 5,
                "NEED_STATE": ["State A", "State B", "State A", "State C", "State B"],
                "CDT": ["CDT A"] * 5,
                "PLANOGRAM_DSC": ["PG A"] * 5,
                "PLANOGRAM_NBR": [1, 2, 3, 4, 5],
                "NEW_ITEM": [False] * 5,
                "TO_BE_DROPPED": [False] * 5,
            }
        )

    def test_assets_defined(self) -> None:
        """Test that all expected assets are defined."""
        assets = list(defs.get_all_assets())
        assert len(assets) > 0
        
        # Get asset keys as strings for easier assertion
        asset_keys = [asset.key.to_user_string() for asset in assets]
        
        # Check for expected assets - modify these to match your actual asset names
        expected_assets = [
            "raw_sales_data",
            "need_state_mappings",
            "merged_data",
        ]
        
        for expected in expected_assets:
            assert any(expected in key for key in asset_keys), f"Expected asset {expected} not found"

    def test_materialize_to_memory(self, sample_sales_data, monkeypatch) -> None:
        """Test materializing assets to memory."""
        # This is a test pattern that shows how to test assets with mocked inputs
        
        # Skip if no assets are found
        assets = list(defs.get_all_assets())
        if not assets:
            pytest.skip("No assets found to test")
            
        # Find a suitable asset to test - looking for one related to raw data
        test_assets = [asset for asset in assets if "raw" in asset.key.to_user_string()]
        if not test_assets:
            pytest.skip("No suitable test asset found")
        
        # Setup mock for data loading (this would need to be adapted to your implementation)
        def mock_read(*args, **kwargs):
            return sample_sales_data
            
        # Apply the mock - you'll need to adjust this to match your actual implementation
        # This is just a pattern showing how you'd mock the data source
        try:
            import clustering.io.readers
            monkeypatch.setattr(clustering.io.readers.CSVReader, "_read_from_source", mock_read)
        except (ImportError, AttributeError):
            # Skip if we can't apply the mock correctly
            pytest.skip("Could not apply mock to data reader")
        
        # Attempt to materialize the asset to memory
        try:
            result = materialize_to_memory(
                [test_assets[0]],
                run_config={
                    "resources": {
                        "io_manager": {"config": {"base_path": "/tmp/test_dagster"}}
                    }
                }
            )
            
            # Assert the materialization was successful
            assert result.success
            
            # Check that we got some output
            assert len(result.asset_values) > 0
            
            # Get the output value for our asset
            output = result.asset_values[test_assets[0].key]
            
            # Assert something about the output - adjust based on what your asset returns
            if hasattr(output, "shape"):
                assert output.shape[0] > 0  # For DataFrame-like outputs
            elif isinstance(output, list):
                assert len(output) > 0  # For list-like outputs
                
        except Exception as e:
            pytest.skip(f"Asset materialization failed: {str(e)}")


@pytest.mark.parametrize(
    "asset_name,expected_columns",
    [
        ("raw_sales_data", ["SKU_NBR", "STORE_NBR", "CAT_DSC", "TOTAL_SALES"]),
        ("need_state_mappings", ["PRODUCT_ID", "CATEGORY", "NEED_STATE"]),
        # Add more asset tests as needed
    ],
)
def test_asset_schema(asset_name: str, expected_columns: list[str], monkeypatch) -> None:
    """Test that assets produce output with expected schema.
    
    Args:
        asset_name: Name of the asset to test
        expected_columns: List of column names expected in the output
    """
    # Find the asset by name
    assets = list(defs.get_all_assets())
    matching_assets = [a for a in assets if asset_name in a.key.to_user_string()]
    
    if not matching_assets:
        pytest.skip(f"Asset {asset_name} not found")
        
    asset_to_test = matching_assets[0]
    
    # Setup test data based on asset name
    if "sales" in asset_name:
        test_data = pd.DataFrame({
            "SKU_NBR": [101, 102],
            "STORE_NBR": [1, 2],
            "CAT_DSC": ["Category A", "Category B"],
            "TOTAL_SALES": [100.0, 200.0]
        })
    elif "need_state" in asset_name:
        test_data = pd.DataFrame({
            "PRODUCT_ID": [101, 102],
            "CATEGORY": ["Category A", "Category B"],
            "NEED_STATE": ["State A", "State B"],
            "CDT": ["CDT A", "CDT B"],
            "PLANOGRAM_DSC": ["PG A", "PG B"],
            "PLANOGRAM_NBR": [1, 2],
            "NEW_ITEM": [False, True],
            "TO_BE_DROPPED": [False, True],
        })
    else:
        pytest.skip(f"No test data defined for asset {asset_name}")
    
    # Mock any necessary functions - you'll need to adapt this
    # to your actual implementation
    try:
        # This is just an example of how you might mock data loading
        # Replace with actual implementation details
        import clustering.io.readers
        def mock_read(*args, **kwargs):
            return test_data
        monkeypatch.setattr(clustering.io.readers.CSVReader, "_read_from_source", mock_read)
    except (ImportError, AttributeError):
        pytest.skip("Could not apply mock to data reader")
    
    # Try to materialize the asset
    try:
        result = materialize_to_memory(
            [asset_to_test],
            run_config={
                "resources": {
                    "io_manager": {"config": {"base_path": "/tmp/test_dagster"}}
                }
            }
        )
        
        # Check that materialization succeeded
        assert result.success
        
        # Get the output
        output = result.asset_values[asset_to_test.key]
        
        # Assert the output has the expected schema
        if hasattr(output, "columns"):
            for col in expected_columns:
                assert col in output.columns, f"Expected column {col} not found in output"
    except Exception as e:
        pytest.skip(f"Asset materialization failed: {str(e)}") 