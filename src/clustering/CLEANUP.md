# Codebase Reorganization Summary

This document describes the reorganization of the clustering package to improve maintainability and organization.

## Files Removed

- `src/clustering/run_dagster.py` - Replaced with `src/clustering/cli/commands.py`
- `src/clustering/dagster_app.py` - Replaced with `src/clustering/dagster/app.py`
- `src/clustering/settings.py` - Replaced with `src/clustering/infra/config.py`
- `src/clustering/config.py` - Replaced with `src/clustering/infra/config.py`
- `src/clustering/scripts.py` - Replaced with `src/clustering/cli/commands.py`
- `src/clustering/io/services.py` - Services moved to dedicated modules:
  - `LoggerService` → `src/clustering/infra/logging.py`
  - `MlflowService` → `src/clustering/infra/mlflow/tracking.py` (removed - not used in actual pipeline)
  - `AlertsService` → `src/clustering/infra/monitoring/alerts.py`

## New Directory Structure

```
clustering/
├── __init__.py                  # Package initialization
├── __main__.py                  # Entry point (calls CLI)
├── CLEANUP.md                   # This file
├── README.md                    # Package documentation
│
├── cli/                         # Command-line interface
│   ├── __init__.py
│   └── commands.py              # CLI commands and utilities
│
├── core/                        # Core domain models
│   ├── __init__.py
│   ├── models.py                # Data models
│   └── schemas.py               # Pydantic schemas
│
├── dagster/                     # Dagster integration
│   ├── __init__.py
│   ├── app.py                   # Web UI application
│   ├── assets/                  # Dagster assets
│   ├── definitions.py           # Dagster definitions
│   ├── resources/               # Dagster resources
│   ├── jobs/                    # Dagster job definitions
│   └── schedules/               # Dagster schedules
│
├── infra/                       # Infrastructure services
│   ├── __init__.py              # Exposes all infrastructure services
│   ├── config.py                # Configuration management
│   ├── logging.py               # Logging service
│   └── monitoring/              # Monitoring services
│       ├── __init__.py
│       └── alerts.py            # Alerting functionality
│
├── io/                          # Input/Output services
│   ├── __init__.py
│   ├── configs.py               # Config parsing helpers (used by tests)
│   ├── datasets.py              # Dataset functionality
│   ├── readers/                 # Data readers
│   │   └── __init__.py          # File readers (CSV, Parquet)
│   └── writers/                 # Data writers
│       └── __init__.py          # File writers (CSV, Parquet)
│
└── utils/                       # Utility functions
    ├── __init__.py
    └── common.py                # Common utilities
```

## Benefits of New Structure

1. **Clearer Separation of Concerns**

   - Infrastructure services (logging, config, monitoring) are grouped in `infra/`
   - Command-line interface is isolated in `cli/`
   - I/O operations are organized in `io/`

2. **Improved Maintainability**

   - Each module has a clear, single responsibility
   - Dependencies between modules are explicit
   - Code is easier to navigate and understand

3. **Better Organization**

   - Related functionality is grouped together
   - File and directory names reflect their purpose
   - Common utility functions are centralized

4. **Enhanced Testability**
   - Components are more isolated
   - Dependencies are easier to mock
   - Clear interfaces between modules

## Running the Application

The application can be run using the new CLI:

```bash
# Run a job
python -m clustering run internal_preprocessing_job --env dev

# Start the web UI
python -m clustering ui --port 3000
```
