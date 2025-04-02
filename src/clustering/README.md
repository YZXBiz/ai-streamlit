# Clustering Package Structure

This document describes the organization of the clustering package.

## Package Structure

```
clustering/
├── __init__.py                  # Package initialization
├── __main__.py                  # Entry point (calls CLI)
│
├── cli/                         # Command-line interface
│   ├── __init__.py
│   └── commands.py              # CLI commands and utilities
│
├── core/                        # Core domain models
│   ├── __init__.py
│   ├── models.py                # Data models and business logic
│   └── schemas.py               # Pydantic schemas and validation
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
├── infra/                       # Infrastructure layer
│   ├── __init__.py
│   ├── config.py                # Configuration handling
│   ├── logging.py               # Logging services
│   └── monitoring/              # Monitoring services
│       ├── __init__.py
│       └── alerts.py            # Alerting functionality
│
├── io/                          # Input/Output services
│   ├── __init__.py
│   ├── datasets.py              # Dataset handling
│   ├── readers/                 # Data readers
│   │   └── __init__.py
│   └── writers/                 # Data writers
│       └── __init__.py
│
└── utils/                       # Utility functions
    └── __init__.py
```

## Components

### CLI

The `cli` module provides command-line interfaces for running jobs and starting the Dagster web UI.

Usage:

```bash
# Run a job
python -m clustering run internal_preprocessing_job --env dev

# Start the Dagster web UI
python -m clustering ui --port 3000
```

### Core

The `core` module contains domain models, schemas, and business logic.

### Infrastructure

The `infra` module contains infrastructure services like logging, monitoring, and configuration management.

### IO

The `io` module contains input/output services for reading and writing data.

### Dagster

The `dagster` module contains Dagster-specific code for defining and running jobs.

## Execution

To run the application:

```bash
# Using the Makefile
make run-internal_preprocessing

# Or directly with Python
python -m clustering run internal_preprocessing_job
```

To start the Dagster web UI:

```bash
python -m clustering ui
```
