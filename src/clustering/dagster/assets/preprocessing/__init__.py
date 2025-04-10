"""Dagster assets for preprocessing pipeline."""

from clustering.dagster.assets.preprocessing.external import (
    external_features_data,
    preprocessed_external_data,
)
from clustering.dagster.assets.preprocessing.internal import (
    internal_normalized_sales_data,
    internal_output_sales_table,
    internal_product_category_mapping,
    internal_raw_sales_data,
    internal_sales_by_category,
    internal_sales_with_categories,
)

__all__ = [
    # Internal preprocessing assets
    "internal_raw_sales_data",
    "internal_product_category_mapping",
    "internal_sales_with_categories",
    "internal_normalized_sales_data",
    "internal_sales_by_category",
    "internal_output_sales_table",
    # External preprocessing assets
    "external_features_data",
    "preprocessed_external_data",
]
