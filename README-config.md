# Configuration System

## Overview

The configuration system for this project uses the Dagster approach exclusively. The Dagster configuration structure is located in:

```
src/clustering/dagster/resources/configs/
```

## Configuration Structure

The configuration structure is organized as follows:

- `src/clustering/dagster/resources/configs/base.yml` - Base configuration with default values
- `src/clustering/dagster/resources/configs/dev.yml` - Development environment configuration
- `src/clustering/dagster/resources/configs/staging.yml` - Staging environment configuration
- `src/clustering/dagster/resources/configs/prod.yml` - Production environment configuration
- `src/clustering/dagster/resources/configs/job_configs/` - Job-specific configuration files

## Configuration Loading Process

1. The base configuration is loaded first (`base.yml`)
2. Environment-specific configuration is applied as an override (`dev.yml`, `staging.yml`, or `prod.yml`)
3. Job-specific configuration is loaded from the appropriate file in `job_configs/`

## Using the Configuration System

All Dagster resources are configured using the environment-specific configuration files, which are loaded through the `load_resource_config()` function in `definitions.py`.

Job-specific configurations are loaded through the `clustering_config` resource, which merges the base, environment, and job-specific configurations.
