# Configuration Files

This directory contains the YAML configuration files used by the Dagster pipeline for different environments.

## Available Configurations

- `dev.yml`: Development environment configuration
- `staging.yml`: Staging environment configuration
- `prod.yml`: Production environment configuration

## Configuration Structure

Each configuration file follows the same structure:

```yaml
# Alerts and logging configuration
alerts:
  enabled: true
  threshold: WARNING
logger:
  level: DEBUG
  sink: logs/dagster_dev.log

# Paths configuration with environment variable substitution
paths:
  base_data_dir: ${env:DATA_DIR,/workspaces/testing-dagster/data}
  internal_data_dir: ${env:INTERNAL_DATA_DIR,/workspaces/testing-dagster/data/internal}
  external_data_dir: ${env:EXTERNAL_DATA_DIR,/workspaces/testing-dagster/data/external}
  merging_data_dir: ${env:MERGING_DATA_DIR,/workspaces/testing-dagster/data/merging}

# Job parameters for feature engineering and model training
job_params:
  # Feature engineering parameters
  normalize: true
  norm_method: "robust"
  # ... other parameters ...

# Data sources (readers) configuration
readers:
  internal_ns_map:
    kind: "CSVReader"
    config:
      path: ${env:INTERNAL_DATA_DIR,/workspaces/testing-dagster/data/internal}/ns_map.csv
  # ... other readers ...

# Data destinations (writers) configuration
writers:
  internal_sales_output:
    kind: "PickleWriter"
    config:
      path: ${env:INTERNAL_DATA_DIR,/workspaces/testing-dagster/data/internal}/sales_by_category.pkl
  # ... other writers ...
```

## Environment Variable Substitution

The configuration files support environment variable substitution with fallback values:

```yaml
path: ${env:VARIABLE_NAME,default_value}
```

This allows for flexible deployment across different environments while maintaining a consistent configuration structure.

## Usage

The configuration files are loaded by the Dagster pipeline based on the environment specified when running the pipeline:

```bash
make run-full ENV=prod  # Uses prod.yml
```

If no environment is specified, the default is `dev`.

## Extending Configuration

When adding new components to the pipeline:

1. Add corresponding configuration entries to all environment files
2. Use environment variables with sensible defaults for paths
3. Document new configuration parameters in comments
4. Keep the structure consistent across all environment files
