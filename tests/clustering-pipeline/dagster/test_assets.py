"""Tests for Dagster assets in the pipeline package."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from dagster import (
    AssetExecutionContext,
    AssetIn,
    AssetKey,
    DagsterInstance,
    asset,
    materialize,
    mem_io_manager,
)

from clustering.pipeline.assets import (
    external_load_raw_data,
    external_preprocess_data,
    external_transform_data,
    internal_load_raw_data,
    internal_preprocess_data,
    internal_transform_data,
    merge_clusters,
)


@pytest.fixture
def mock_asset_context() -> AssetExecutionContext:
    """Mock asset execution context for testing."""
    # This is a simplified version - in real tests you might want to
    # create a more complete context with run_id, op_name, etc.
    return AssetExecutionContext(resources={}, asset_key=AssetKey("test_asset"))


@pytest.fixture
def internal_raw_data() -> pd.DataFrame:
    """Sample internal raw data for testing."""
    return pd.DataFrame(
        {
            "SKU_NBR": [1001, 1002, 1003, 1001, 1002, 1003],
            "STORE_NBR": [101, 101, 101, 102, 102, 102],
            "CAT_DSC": ["Health", "Beauty", "Grocery", "Health", "Beauty", "Grocery"],
            "TOTAL_SALES": [1500.50, 2200.75, 3100.25, 1400.50, 2100.75, 3000.25],
        }
    )


@pytest.fixture
def external_raw_data() -> pd.DataFrame:
    """Sample external raw data for testing."""
    return pd.DataFrame(
        {
            "product_id": [1001, 1002, 1003],
            "category": ["Health", "Beauty", "Grocery"],
            "need_state": ["Pain Relief", "Moisturizing", "Snacks"],
            "cdt": ["Tablets", "Lotion", "Chips"],
            "planogram_desc": ["PAIN RELIEF", "SKIN CARE", "SNACKS"],
            "new_item": [False, True, False],
        }
    )


@pytest.fixture
def internal_preprocessed_data() -> pd.DataFrame:
    """Sample internal preprocessed data for testing."""
    return pd.DataFrame(
        {
            "SKU_NBR": [1001, 1002, 1003, 1001, 1002, 1003],
            "STORE_NBR": [101, 101, 101, 102, 102, 102],
            "CAT_DSC": ["Health", "Beauty", "Grocery", "Health", "Beauty", "Grocery"],
            "TOTAL_SALES": [1500.50, 2200.75, 3100.25, 1400.50, 2100.75, 3000.25],
            "NORMALIZED_SALES": [0.48, 0.71, 1.0, 0.47, 0.70, 1.0],
        }
    )


@pytest.fixture
def external_preprocessed_data() -> pd.DataFrame:
    """Sample external preprocessed data for testing."""
    return pd.DataFrame(
        {
            "product_id": [1001, 1002, 1003],
            "category": ["Health", "Beauty", "Grocery"],
            "need_state": ["Pain Relief", "Moisturizing", "Snacks"],
            "importance_score": [0.8, 0.7, 0.9],
        }
    )


@pytest.fixture
def internal_transformed_data() -> pd.DataFrame:
    """Sample internal transformed data for testing."""
    return pd.DataFrame(
        {
            "SKU_NBR": [1001, 1002, 1003],
            "STORE_NBR": [101, 101, 101],
            "CAT_DSC": ["Health", "Beauty", "Grocery"],
            "NEED_STATE": ["Pain Relief", "Moisturizing", "Snacks"],
            "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
            "NORMALIZED_SALES": [0.48, 0.71, 1.0],
            "CLUSTER": [0, 1, 2],
        }
    )


@pytest.fixture
def external_transformed_data() -> pd.DataFrame:
    """Sample external transformed data for testing."""
    return pd.DataFrame(
        {
            "product_id": [1001, 1002, 1003],
            "category": ["Health", "Beauty", "Grocery"],
            "need_state": ["Pain Relief", "Moisturizing", "Snacks"],
            "importance_score": [0.8, 0.7, 0.9],
            "cluster": [0, 1, 2],
        }
    )


@pytest.fixture
def in_memory_dagster_instance() -> DagsterInstance:
    """Create an in-memory Dagster instance for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        instance = DagsterInstance.ephemeral(tempdir=temp_dir)
        yield instance


class TestInternalAssets:
    """Tests for internal data assets."""

    def test_internal_load_raw_data(self, mock_asset_context: AssetExecutionContext) -> None:
        """Test loading raw internal data."""
        # Create a temporary CSV file with test data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data
            data = pd.DataFrame(
                {
                    "SKU_NBR": [1001, 1002, 1003],
                    "STORE_NBR": [101, 101, 101],
                    "CAT_DSC": ["Health", "Beauty", "Grocery"],
                    "TOTAL_SALES": [1500.50, 2200.75, 3100.25],
                }
            )

            # Write data to CSV
            data.to_csv(temp_path, index=False)

        try:
            # Mock the config
            config = {"data_path": str(temp_path)}

            # Execute the asset
            result = internal_load_raw_data(mock_asset_context, config)

            # Validate result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "SKU_NBR" in result.columns
            assert "STORE_NBR" in result.columns
            assert "CAT_DSC" in result.columns
            assert "TOTAL_SALES" in result.columns

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_internal_preprocess_data(
        self, mock_asset_context: AssetExecutionContext, internal_raw_data: pd.DataFrame
    ) -> None:
        """Test preprocessing internal data."""
        # Mock the config
        config = {"normalize_sales": True}

        # Execute the asset
        result = internal_preprocess_data(mock_asset_context, internal_raw_data, config)

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # Should maintain all rows
        assert "NORMALIZED_SALES" in result.columns  # Should add this column
        assert all(0 <= val <= 1 for val in result["NORMALIZED_SALES"])  # Normalized values

    def test_internal_transform_data(
        self,
        mock_asset_context: AssetExecutionContext,
        internal_preprocessed_data: pd.DataFrame,
        external_preprocessed_data: pd.DataFrame,
    ) -> None:
        """Test transforming internal data."""
        # Mock the config
        config = {"algorithm": "kmeans", "n_clusters": 3, "random_state": 42}

        # Execute the asset
        result = internal_transform_data(
            mock_asset_context, internal_preprocessed_data, external_preprocessed_data, config
        )

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert "CLUSTER" in result.columns  # Should add this column
        assert set(result["CLUSTER"].unique()) <= {
            0,
            1,
            2,
        }  # Cluster values should be in expected range


class TestExternalAssets:
    """Tests for external data assets."""

    def test_external_load_raw_data(self, mock_asset_context: AssetExecutionContext) -> None:
        """Test loading raw external data."""
        # Create a temporary CSV file with test data
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp_path = Path(temp.name)

            # Create test data
            data = pd.DataFrame(
                {
                    "product_id": [1001, 1002, 1003],
                    "category": ["Health", "Beauty", "Grocery"],
                    "need_state": ["Pain Relief", "Moisturizing", "Snacks"],
                    "cdt": ["Tablets", "Lotion", "Chips"],
                    "planogram_desc": ["PAIN RELIEF", "SKIN CARE", "SNACKS"],
                    "new_item": [False, True, False],
                }
            )

            # Write data to CSV
            data.to_csv(temp_path, index=False)

        try:
            # Mock the config
            config = {"data_path": str(temp_path)}

            # Execute the asset
            result = external_load_raw_data(mock_asset_context, config)

            # Validate result
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3
            assert "product_id" in result.columns
            assert "category" in result.columns
            assert "need_state" in result.columns

        finally:
            # Cleanup
            if temp_path.exists():
                os.unlink(temp_path)

    def test_external_preprocess_data(
        self, mock_asset_context: AssetExecutionContext, external_raw_data: pd.DataFrame
    ) -> None:
        """Test preprocessing external data."""
        # Mock the config
        config = {"calculate_importance": True}

        # Execute the asset
        result = external_preprocess_data(mock_asset_context, external_raw_data, config)

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # Should maintain all rows
        assert "importance_score" in result.columns  # Should add this column

    def test_external_transform_data(
        self, mock_asset_context: AssetExecutionContext, external_preprocessed_data: pd.DataFrame
    ) -> None:
        """Test transforming external data."""
        # Mock the config
        config = {"algorithm": "kmeans", "n_clusters": 3, "random_state": 42}

        # Execute the asset
        result = external_transform_data(mock_asset_context, external_preprocessed_data, config)

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert "cluster" in result.columns  # Should add this column
        assert set(result["cluster"].unique()) <= {
            0,
            1,
            2,
        }  # Cluster values should be in expected range


class TestMergingAssets:
    """Tests for merging cluster assets."""

    def test_merge_clusters(
        self,
        mock_asset_context: AssetExecutionContext,
        internal_transformed_data: pd.DataFrame,
        external_transformed_data: pd.DataFrame,
    ) -> None:
        """Test merging clusters from internal and external data."""
        # Mock the config
        config = {"matching_threshold": 0.7}

        # Execute the asset
        result = merge_clusters(
            mock_asset_context, internal_transformed_data, external_transformed_data, config
        )

        # Validate result
        assert isinstance(result, pd.DataFrame)
        assert "internal_cluster" in result.columns
        assert "external_cluster" in result.columns
        assert "merged_cluster" in result.columns
        assert "match_score" in result.columns
        assert all(
            0 <= score <= 1 for score in result["match_score"]
        )  # Match scores should be in [0,1]


class TestAssetMaterialization:
    """Tests for materializing assets with Dagster."""

    def test_asset_materialization(self, in_memory_dagster_instance: DagsterInstance) -> None:
        """Test that assets can be materialized successfully."""
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory():
            # Define test assets
            @asset
            def test_input_asset() -> pd.DataFrame:
                """Generate test input data."""
                return pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

            @asset(ins={"input_data": AssetIn("test_input_asset")})
            def test_output_asset(input_data: pd.DataFrame) -> pd.DataFrame:
                """Process input data."""
                return input_data.assign(processed=input_data["value"] * 2)

            # Materialize the assets
            result = materialize(
                [test_input_asset, test_output_asset],
                instance=in_memory_dagster_instance,
                resources={"io_manager": mem_io_manager},
            )

            # Verify materialization was successful
            assert result.success
            assert len(result.asset_materializations_for_node("test_input_asset")) == 1
            assert len(result.asset_materializations_for_node("test_output_asset")) == 1
