# Configuration System

This directory contains all configuration files for the Dagster pipeline implementation. The configuration system uses a layered approach with environment-specific overrides.

## Directory Structure

- `base.yml` - Base configuration with default values
- `dev.yml` - Development environment configuration
- `staging.yml` - Staging environment configuration
- `prod.yml` - Production environment configuration
- `job_configs/` - Job-specific configuration files

## Configuration Loading Process

1. The base configuration is loaded first (`base.yml`)
2. Environment-specific configuration is applied as an override (`dev.yml`, `staging.yml`, or `prod.yml`)
3. Job-specific configuration is loaded from the appropriate file in `job_configs/`

## Configuration Reference

### Resource Configuration

Resource configuration is specified in the environment-specific files and includes:

- IO Manager configuration
- Logger configuration
- Alerts configuration
- Data reader/writer configuration

### Job Configuration

Job configuration is specified in job-specific files and includes:

- Clustering parameters
- Preprocessing parameters
- Input/output specifications
- Algorithm settings
- Feature weights

## Usage

The configuration is loaded through `load_resource_config()` in `definitions.py` and used to configure Dagster resources through the `get_resources_by_env()` function.

Job configurations are loaded through the `clustering_config` resource, which merges the base, environment, and job-specific configurations.

## Legacy Configuration

The project previously used a different configuration system with the root-level `/configs` directory. That system is now deprecated and not used in the current implementation.
