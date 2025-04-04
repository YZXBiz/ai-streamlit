# Configuration System Changes

## Overview

The configuration system for this project has been consolidated to use the Dagster approach exclusively. The Dagster approach uses a simpler, more direct configuration structure located in:

```
src/clustering/dagster/resources/configs/
```

The previous configuration system defined in `src/clustering/io/config_parser.py` (using the `ConfigManager` class) is now deprecated and not used in the current implementation.

## Migration

A migration script has been provided to help transition from the old configuration structure to the new one:

```bash
python scripts/migrate_configs.py
```

This script will copy relevant configuration files from the old structure to the new Dagster structure but won't delete the original files. Once you've verified that everything works correctly, you can safely remove the old configuration files.

## New Configuration Structure

The new configuration structure is organized as follows:

- `src/clustering/dagster/resources/configs/base.yml` - Base configuration with default values
- `src/clustering/dagster/resources/configs/dev.yml` - Development environment configuration
- `src/clustering/dagster/resources/configs/staging.yml` - Staging environment configuration
- `src/clustering/dagster/resources/configs/prod.yml` - Production environment configuration
- `src/clustering/dagster/resources/configs/job_configs/` - Job-specific configuration files

## Configuration Loading Process

1. The base configuration is loaded first (`base.yml`)
2. Environment-specific configuration is applied as an override (`dev.yml`, `staging.yml`, or `prod.yml`)
3. Job-specific configuration is loaded from the appropriate file in `job_configs/`

## Legacy Configuration

The root-level `/configs` directory is no longer used and can be removed after migration. The configuration system defined in `src/clustering/io/config_parser.py` has been marked as deprecated but preserved for reference.

## Using the Configuration System

All Dagster resources are configured using the environment-specific configuration files, which are loaded through the `load_resource_config()` function in `definitions.py`.

Job-specific configurations are loaded through the `clustering_config` resource, which merges the base, environment, and job-specific configurations.
