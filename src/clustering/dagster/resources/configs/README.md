# Configuration Files

This directory contains the configuration files for various environments in the clustering pipeline.

## Structure

The system uses a simplified configuration approach with a single file per environment:

- `dev.yml` - Development environment configuration
- `staging.yml` - Staging environment configuration
- `prod.yml` - Production environment configuration

## Configuration Structure

Each configuration file has the following sections:

1. **Logger** - Configuration for the logger service

   ```yaml
   logger:
     level: INFO
     sink: logs/dagster_dev.log
   ```

2. **Alerts** - Configuration for the alerts service

   ```yaml
   alerts:
     enabled: true
     threshold: WARNING
   ```

3. **Job Parameters** - Parameters for all job types

   ```yaml
   job_params:
     # Common parameters
     algorithm: "kmeans"
     normalize: true

     # Algorithm specific parameters
     kmeans:
       n_clusters: 5
       random_state: 42

     # Evaluation parameters
     evaluation:
       metrics:
         - "silhouette_score"
   ```

4. **Readers/Writers** - Configuration for data sources and destinations

   ```yaml
   readers:
     external_sales:
       kind: "SnowflakeReader"
       config:
         query: SELECT * FROM DEV_CLUSTERING_DB.RAW.EXTERNAL_SALES

   writers:
     internal_clusters_output:
       kind: "SnowflakeWriter"
       config:
         table: DEV_CLUSTERING_DB.PROCESSED.INTERNAL_CLUSTERS
         database: DEV_CLUSTERING_DB
         schema: PROCESSED
   ```

## Available Reader/Writer Parameters

### Readers

#### SnowflakeReader

- **Required Parameters:**
  - `query`: SQL query to execute
- **Optional Parameters:**
  - `use_cache`: Whether to cache query results (default: `true`)
  - `cache_file`: Path to the cache file (default: `"cache/snowflake_cache.duckdb"`)
  - `pkb_path`: Path to the private key file (default: `"creds/pkb.pkl"`)
  - `creds_path`: Path to the credentials JSON file (default: `"creds/sf_creds.json"`)
  - `limit`: Maximum number of rows to read (default: `None`)

#### BlobReader

- **Required Parameters:**
  - `blob_name`: Path to the blob in storage
- **Optional Parameters:**
  - `max_concurrency`: Maximum concurrency for downloads (default: `8`)
  - `limit`: Maximum number of rows to read (default: `None`)

#### Other Available Readers

- **CSVReader**: Reads from CSV files (requires `path` parameter)
- **ParquetReader**: Reads from Parquet files (requires `path` parameter)
- **ExcelReader**: Reads from Excel files (requires `path` parameter)
- **PickleReader**: Reads from Pickle files (requires `path` parameter)

### Writers

#### SnowflakeWriter

- **Required Parameters:**
  - `table`: Target table name
  - `database`: Target database
  - `schema`: Target schema
- **Optional Parameters:**
  - `auto_create_table`: Create table if it doesn't exist (default: `true`)
  - `overwrite`: Overwrite existing data (default: `true`)
  - `pkb_path`: Path to the private key file (default: `"creds/pkb.pkl"`)
  - `creds_path`: Path to the credentials JSON file (default: `"creds/sf_creds.json"`)

#### BlobWriter

- **Required Parameters:**
  - `blob_name`: Target blob path
- **Optional Parameters:**
  - `overwrite`: Overwrite existing data (default: `true`)
  - `max_concurrency`: Maximum concurrency for uploads (default: `8`)

#### Other Available Writers

- **CSVWriter**: Writes to CSV files (requires `path` parameter)
- **ParquetWriter**: Writes to Parquet files (requires `path` parameter)
- **ExcelWriter**: Writes to Excel files (requires `path` parameter)
- **PickleWriter**: Writes to Pickle files (requires `path` parameter)

## Using the Configuration

The configuration files are loaded by the Dagster pipeline at startup. The environment to use is
specified when creating resources:

```python
resources = get_resources_by_env("dev")  # Use dev.yml
```

Within assets, you can access configuration values through the `config` resource:

```python
# Get config from resources
config = context.resources.config

# Access parameters directly from job_params
job_params = config.job_params
algorithm = getattr(job_params, "algorithm", "kmeans")
```
