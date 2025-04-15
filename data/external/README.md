# External Data Directory

This directory contains all external data sources used in the store clustering pipeline.

## Key Files

- `placerai.csv`: External demographic and geographic data from PlaceAI
- `processed_external_data.pkl`: Processed external data ready for feature engineering
- `clustering_models.pkl`: Trained clustering models based on external data
- `cluster_assignments.pkl`: Cluster assignments for stores based on external data

## Data Flow

1. Raw external data from `placerai.csv` is loaded
2. Data is processed and saved as `processed_external_data.pkl`
3. Feature engineering is applied (transformation happens in memory)
4. Models are trained and saved as `clustering_models.pkl`
5. Stores are assigned to clusters in `cluster_assignments.pkl`

## Environment Variable

This directory is referenced by the environment variable `EXTERNAL_DATA_DIR` with default value `/workspaces/testing-dagster/data/external`
