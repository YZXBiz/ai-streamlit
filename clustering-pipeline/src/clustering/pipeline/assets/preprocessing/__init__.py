"""Preprocessing assets for the clustering pipeline."""

from pipeline.assets.preprocessing.external import (
    external_features_data,
    preprocessed_external_data,
)

from pipeline.assets.preprocessing.internal import (
    internal_normalized_sales_data,
    internal_output_sales_table,
    internal_product_category_mapping,
    internal_raw_sales_data,
    internal_sales_by_category,
    internal_sales_with_categories,
)

__all__ = [
    # External preprocessing assets
    "external_features_data",
    "preprocessed_external_data",
    # Internal preprocessing assets
    "internal_normalized_sales_data",
    "internal_output_sales_table",
    "internal_product_category_mapping",
    "internal_raw_sales_data",
    "internal_sales_by_category",
    "internal_sales_with_categories",
]
