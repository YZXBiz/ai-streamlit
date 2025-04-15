# Raw Data Directory

This directory contains the original, unprocessed data files used as inputs to the clustering pipeline.

## Expected Files

- `ns_map_raw.csv`: Raw store category mapping before preprocessing
- `ns_sales_raw.csv`: Raw sales transaction data before aggregation
- `placerai_raw.csv`: Raw PlaceAI external data before processing

## Data Sources

- **Internal Sales Data**: Extracts from enterprise data warehouse
- **Store Categories**: Store categorization from merchandising systems
- **External Data**: Geographic and demographic data from third-party providers

## Usage

These raw files serve as the starting point for the pipeline. They should not be modified manually after placement in this directory. The pipeline reads these files, applies transformations, and stores the results in the appropriate directories:

- Internal data → `../internal/`
- External data → `../external/`

## Notes

- Data files should be placed here before running the pipeline
- Files should be in the expected format (CSV with proper headers)
- Make sure all required files are present before starting the pipeline
- This directory is for initial data only; processed data will be saved elsewhere
