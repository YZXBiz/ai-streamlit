# Internal Data Directory

This directory contains all internal store data used for clustering analysis.

## Key Files

- `ns_map.csv`: Store category mapping file containing store numbers and their associated categories
- `ns_sales.csv`: Store sales data with transaction information
- `sales_by_category.pkl`: Processed sales data aggregated by category
- `sales_by_category_percent.pkl`: Percent sales by category for normalization
- `engineered_features.pkl`: Feature engineered dataset ready for model training
- `clustering_models.pkl`: Trained clustering models with metadata
- `cluster_assignments.pkl`: Resulting cluster assignments for stores

## Data Flow

1. Raw `ns_map.csv` and `ns_sales.csv` files are processed
2. Sales are aggregated by category and saved as `sales_by_category.pkl`
3. Feature engineering creates `engineered_features.pkl`
4. Model training produces `clustering_models.pkl`
5. The final step generates `cluster_assignments.pkl`

## Environment Variable

This directory is referenced by the environment variable `INTERNAL_DATA_DIR` with default value `/workspaces/testing-dagster/data/internal` 