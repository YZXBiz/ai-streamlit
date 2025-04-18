"""Tests for the schema validation module."""

import pandas as pd
import polars as pl
import pytest
from pandera.errors import SchemaError

from clustering.core.schemas import (
    Category,
    DistributedDataSchema,
    MergedDataSchema,
    NSMappingSchema,
    NeedState,
    SalesSchema,
    Schema,
)


@pytest.fixture
def sample_data() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"]
    })


class TestSchema:
    """Tests for the base Schema class."""

    def test_check_method_pandas(self) -> None:
        """Test the check method with pandas DataFrame."""
        # Create a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Valid data
        valid_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = TestSchema.check(valid_data)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["col1", "col2"]

    def test_check_method_polars(self) -> None:
        """Test the check method with polars DataFrame."""
        # Create a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Valid data
        valid_data = pl.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        result = TestSchema.check(valid_data)

        assert isinstance(result, pl.DataFrame)
        assert list(result.columns) == ["col1", "col2"]

    def test_invalid_data(self) -> None:
        """Test that the schema raises an error for invalid data."""
        # Create a simple schema for testing
        class TestSchema(Schema):
            col1: int
            col2: str

        # Invalid data (wrong type)
        invalid_data = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1, 2, 3]})

        with pytest.raises(SchemaError):
            TestSchema.check(invalid_data)

    def test_has_duplicates_method(self) -> None:
        """Test the _has_duplicates method in the base Schema class."""
        # Test with pandas DataFrame
        df_pd = pd.DataFrame({
            "col1": [1, 2, 2, 3],
            "col2": ["a", "b", "b", "c"]
        })
        assert Schema._has_duplicates(df_pd, ["col1"])
        assert Schema._has_duplicates(df_pd, ["col1", "col2"])
        assert not Schema._has_duplicates(df_pd, ["col1", "col2", "non_existent"])  # Missing col handling

        # Test with polars DataFrame
        df_pl = pl.DataFrame({
            "col1": [1, 2, 2, 3],
            "col2": ["a", "b", "b", "c"]
        })
        assert Schema._has_duplicates(df_pl, ["col1"])
        assert Schema._has_duplicates(df_pl, ["col1", "col2"])


class TestCategoryEnum:
    """Tests for the Category enum."""
    
    def test_valid_categories(self) -> None:
        """Test accessing valid category values."""
        assert Category.HEALTH.value == "Health"
        assert Category.GROCERY.value == "Grocery"
        assert Category.BEAUTY.value == "Beauty"
        assert Category.BEVERAGES.value == "Beverages"
    
    def test_case_insensitive_match(self) -> None:
        """Test that categories can be matched case-insensitively."""
        assert Category("health") == Category.HEALTH
        assert Category("GROCERY") == Category.GROCERY
        assert Category("Beauty") == Category.BEAUTY
    
    def test_unknown_category(self) -> None:
        """Test handling of unknown categories."""
        unknown = Category("unknown_category")
        assert unknown == Category.OTHER
        assert unknown.value == "Other"


class TestNeedStateEnum:
    """Tests for the NeedState enum."""
    
    def test_valid_need_states(self) -> None:
        """Test accessing valid need state values."""
        assert NeedState.BASIC_ESSENTIALS.value == "Basic Essentials"
        assert NeedState.HEALTH_MANAGEMENT.value == "Health Management"
        assert NeedState.PERSONAL_WELLNESS.value == "Personal Wellness"
    
    def test_case_insensitive_match(self) -> None:
        """Test that need states can be matched case-insensitively."""
        assert NeedState("basic essentials") == NeedState.BASIC_ESSENTIALS
        assert NeedState("HEALTH MANAGEMENT") == NeedState.HEALTH_MANAGEMENT
        assert NeedState("Personal Wellness") == NeedState.PERSONAL_WELLNESS
    
    def test_unknown_need_state(self) -> None:
        """Test handling of unknown need states."""
        unknown = NeedState("unknown_need_state")
        assert unknown == NeedState.OTHER
        assert unknown.value == "Other"


class TestSalesSchema:
    """Tests for SalesSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid sales data."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        validated_data = SalesSchema.check(data)
        assert len(validated_data) == 3

    def test_polars_dataframe(self) -> None:
        """Test validation with polars DataFrame."""
        # Create valid data
        valid_data = pl.DataFrame({
            "SKU_NBR": [1001, 1002, 1003],
            "STORE_NBR": [501, 502, 503],
            "CAT_DSC": ["Health", "Beauty", "Grocery"],
            "TOTAL_SALES": [1500.50, 2200.75, 3100.25]
        })
        
        # Validate data
        result = SalesSchema.check(valid_data)
        
        # Assert validation passed
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_data_coercion(self) -> None:
        """Test coercion of data types."""
        # Strings that should be coerced to numbers
        data = pd.DataFrame(
            {
                "SKU_NBR": ["101", "102", "103"],
                "STORE_NBR": ["1", "2", "3"],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "TOTAL_SALES": ["100.0", "200.0", "300.0"],
            }
        )

        validated_data = SalesSchema.check(data)
        assert isinstance(validated_data["SKU_NBR"].iloc[0], int)
        assert isinstance(validated_data["TOTAL_SALES"].iloc[0], float)

    def test_invalid_data_negative_values(self) -> None:
        """Test validation fails with negative sales values."""
        # Create invalid data with negative sales
        invalid_data = pl.DataFrame({
            "SKU_NBR": [1001, 1002, 1003],
            "STORE_NBR": [501, 502, 503],
            "CAT_DSC": ["Health", "Beauty", "Grocery"],
            "TOTAL_SALES": [1500.50, -2200.75, 3100.25]  # Negative value
        })
        
        # Validate data - should raise SchemaError
        with pytest.raises(SchemaError):
            SalesSchema.check(invalid_data)

    def test_missing_required_column(self) -> None:
        """Test validation fails with missing columns."""
        # Missing TOTAL_SALES column
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
            }
        )

        with pytest.raises(SchemaError):
            SalesSchema.check(data)
            
    def test_invalid_category(self) -> None:
        """Test validation fails with invalid category values."""
        invalid_data = pd.DataFrame({
            "SKU_NBR": [101, 102, 103],
            "STORE_NBR": [1, 2, 3],
            "CAT_DSC": ["Health", "InvalidCategory", "Grocery"],  # Invalid category
            "TOTAL_SALES": [100.0, 200.0, 300.0],
        })
        
        # The schema should still validate since we allow unknown categories to be mapped to OTHER
        validated_data = SalesSchema.check(invalid_data)
        assert len(validated_data) == 3
    
    def test_duplicate_entries(self) -> None:
        """Test validation fails with duplicate key entries."""
        duplicate_data = pd.DataFrame({
            "SKU_NBR": [101, 102, 101],  # Duplicate SKU_NBR + STORE_NBR combination
            "STORE_NBR": [1, 2, 1],      # Duplicate
            "CAT_DSC": ["Health", "Beauty", "Health"],
            "TOTAL_SALES": [100.0, 200.0, 300.0],
        })
        
        # Should raise SchemaError due to duplicates
        with pytest.raises(SchemaError):
            SalesSchema.check(duplicate_data)


class TestNSMappingSchema:
    """Tests for NSMappingSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid need state mapping data."""
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": ["Attr 1", "Attr 2", None],
                "ATTRIBUTE_2": ["Attr 1", None, "Attr 3"],
                "ATTRIBUTE_3": [None, "Attr 2", "Attr 3"],
                "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3"],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [True, False, True],
                "TO_BE_DROPPED": [False, True, False],
            }
        )

        validated_data = NSMappingSchema.check(data)
        assert len(validated_data) == 3

    def test_data_coercion(self) -> None:
        """Test coercion of boolean values."""
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": ["Attr 1", "Attr 2", None],
                "ATTRIBUTE_2": ["Attr 1", None, "Attr 3"],
                "ATTRIBUTE_3": [None, "Attr 2", "Attr 3"],
                "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3"],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [1, 0, 1],  # Integers should be coerced to booleans
                "TO_BE_DROPPED": [0, 1, 0],  # Integers should be coerced to booleans
            }
        )

        validated_data = NSMappingSchema.check(data)
        assert all(isinstance(x, bool) for x in validated_data["NEW_ITEM"])
        assert all(isinstance(x, bool) for x in validated_data["TO_BE_DROPPED"])

    def test_nullable_attributes(self) -> None:
        """Test that attribute columns can contain null values."""
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": [None, None, None],  # All nulls
                "ATTRIBUTE_2": [None, None, None],  # All nulls
                "ATTRIBUTE_3": [None, None, None],  # All nulls
                "ATTRIBUTE_4": [None, None, None],  # All nulls
                "ATTRIBUTE_5": [None, None, None],  # All nulls
                "ATTRIBUTE_6": [None, None, None],  # All nulls
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [True, False, True],
                "TO_BE_DROPPED": [False, True, False],
            }
        )

        validated_data = NSMappingSchema.check(data)
        assert len(validated_data) == 3
        
    def test_invalid_category_values(self) -> None:
        """Test validation behavior with invalid category values."""
        # Invalid category should still be accepted and mapped to 'Other'
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Health", "InvalidCategory", "Grocery"],  # Invalid category
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": ["Attr 1", "Attr 2", None],
                "ATTRIBUTE_2": ["Attr 1", None, "Attr 3"],
                "ATTRIBUTE_3": [None, "Attr 2", "Attr 3"],
                "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3"],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [True, False, True],
                "TO_BE_DROPPED": [False, True, False],
            }
        )
        
        validated_data = NSMappingSchema.check(data)
        assert len(validated_data) == 3
    
    def test_invalid_need_state_values(self) -> None:
        """Test validation behavior with invalid need state values."""
        # Invalid need state should still be accepted and mapped to 'Other'
        data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 103],
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "InvalidNeedState", "Health Management"],  # Invalid
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": ["Attr 1", "Attr 2", None],
                "ATTRIBUTE_2": ["Attr 1", None, "Attr 3"],
                "ATTRIBUTE_3": [None, "Attr 2", "Attr 3"],
                "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3"],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [True, False, True],
                "TO_BE_DROPPED": [False, True, False],
            }
        )
        
        validated_data = NSMappingSchema.check(data)
        assert len(validated_data) == 3
    
    def test_duplicate_product_ids(self) -> None:
        """Test validation fails with duplicate product IDs."""
        duplicate_data = pd.DataFrame(
            {
                "PRODUCT_ID": [101, 102, 101],  # Duplicate product ID
                "CATEGORY": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
                "CDT": ["CDT A", "CDT B", "CDT C"],
                "ATTRIBUTE_1": ["Attr 1", "Attr 2", None],
                "ATTRIBUTE_2": ["Attr 1", None, "Attr 3"],
                "ATTRIBUTE_3": [None, "Attr 2", "Attr 3"],
                "ATTRIBUTE_4": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_5": ["Attr 1", "Attr 2", "Attr 3"],
                "ATTRIBUTE_6": ["Attr 1", "Attr 2", "Attr 3"],
                "PLANOGRAM_DSC": ["PG A", "PG B", "PG C"],
                "PLANOGRAM_NBR": [1, 2, 3],
                "NEW_ITEM": [True, False, True],
                "TO_BE_DROPPED": [False, True, False],
            }
        )
        
        # Should raise SchemaError due to duplicates
        with pytest.raises(SchemaError):
            NSMappingSchema.check(duplicate_data)


class TestMergedDataSchema:
    """Tests for MergedDataSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid merged data."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        validated_data = MergedDataSchema.check(data)
        assert len(validated_data) == 3

    def test_polars_dataframe(self) -> None:
        """Test validation with polars DataFrame."""
        valid_data = pl.DataFrame({
            "SKU_NBR": [1001, 1002, 1003],
            "STORE_NBR": [501, 502, 503],
            "CAT_DSC": ["Health", "Beauty", "Grocery"],
            "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
            "TOTAL_SALES": [1500.50, 2200.75, 3100.25]
        })
        
        result = MergedDataSchema.check(valid_data)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3
    
    def test_invalid_category_and_need_state(self) -> None:
        """Test validation behavior with invalid categories and need states."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Health", "InvalidCategory", "Grocery"],  # Invalid category
                "NEED_STATE": ["Basic Essentials", "InvalidNeedState", "Health Management"],  # Invalid need state
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )
        
        # These should be mapped to "Other" by the enum's _missing_ method
        validated_data = MergedDataSchema.check(data)
        assert len(validated_data) == 3
    
    def test_duplicate_entries(self) -> None:
        """Test validation fails with duplicate key entries."""
        duplicate_data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 101],  # Duplicate SKU_NBR + STORE_NBR + NEED_STATE
                "STORE_NBR": [1, 2, 1],      # Duplicate
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Basic Essentials"],  # Duplicate
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )
        
        # Should raise SchemaError due to duplicates
        with pytest.raises(SchemaError):
            MergedDataSchema.check(duplicate_data)


class TestDistributedDataSchema:
    """Tests for DistributedDataSchema."""

    def test_valid_data(self) -> None:
        """Test validation of valid distributed data."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 103],
                "STORE_NBR": [1, 2, 3],
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )

        validated_data = DistributedDataSchema.check(data)
        assert len(validated_data) == 3

    def test_polars_dataframe(self) -> None:
        """Test validation with polars DataFrame."""
        valid_data = pl.DataFrame({
            "SKU_NBR": [1001, 1002, 1003],
            "STORE_NBR": [501, 502, 503],
            "CAT_DSC": ["Health", "Beauty", "Grocery"],
            "NEED_STATE": ["Basic Essentials", "Beauty Care", "Health Management"],
            "TOTAL_SALES": [1500.50, 2200.75, 3100.25]
        })
        
        result = DistributedDataSchema.check(valid_data)
        
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3

    def test_empty_dataframe(self) -> None:
        """Test validation fails with empty DataFrame."""
        data = pd.DataFrame(
            {
                "SKU_NBR": [],
                "STORE_NBR": [],
                "CAT_DSC": [],
                "NEED_STATE": [],
                "TOTAL_SALES": [],
            }
        )

        with pytest.raises(SchemaError):
            DistributedDataSchema.check(data)
    
    def test_duplicate_entries(self) -> None:
        """Test validation fails with duplicate key entries."""
        duplicate_data = pd.DataFrame(
            {
                "SKU_NBR": [101, 102, 101],  # Duplicate SKU_NBR + STORE_NBR + NEED_STATE
                "STORE_NBR": [1, 2, 1],      # Duplicate
                "CAT_DSC": ["Health", "Beauty", "Grocery"],
                "NEED_STATE": ["Basic Essentials", "Beauty Care", "Basic Essentials"],  # Duplicate
                "TOTAL_SALES": [100.0, 200.0, 300.0],
            }
        )
        
        # Should raise SchemaError due to duplicates
        with pytest.raises(SchemaError):
            DistributedDataSchema.check(duplicate_data) 