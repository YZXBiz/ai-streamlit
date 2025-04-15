# Codebase Overview

This document provides a detailed overview of the repository structure and its components, based on automated analysis.

## 1. Project Structure

The repository follows a standard Python project layout, incorporating tools for dependency management, testing, containerization, and code quality.

```
.
├── .devcontainer/       # VS Code Development Container configuration
├── .git/                # Git repository data
├── .github/             # GitHub Actions workflows (Assumed, common practice)
├── .ruff_cache/         # Cache for Ruff linter
├── .venv/               # Virtual environment (managed by uv)
├── data/                # Likely holds raw or intermediate data (content not inspected)
├── dagster_home/        # Dagster instance home directory (logs, schedules, etc.)
├── docs/                # Project documentation (content not inspected)
├── documents/           # Unclear purpose, potentially more documentation or data (content not inspected)
├── experiments/         # Likely contains experimental code or notebooks (content not inspected)
├── logs/                # General application logs (distinct from Dagster execution logs)
├── src/                 # Main source code directory
│   └── clustering/      # Core Python package for the clustering application
├── storage/             # Default location for Dagster asset materializations (FilesystemIOManager base_dir)
├── tests/               # Unit and integration tests (content not inspected)
├── .cursor/             # Cursor IDE metadata
├── .env                 # Environment variables (loaded potentially by Dagster/Hydra)
├── .env.example         # Example environment file
├── .gitignore           # Specifies intentionally untracked files for Git
├── .pre-commit-config.yaml # Configuration for pre-commit hooks
├── .python-version      # Specifies the Python version (e.g., for pyenv)
├── CODEBASE_OVERVIEW.md # This file
├── Dockerfile           # Defines the Docker image for the application
├── Makefile             # Contains make targets for common development tasks (lint, test, build, etc.)
├── README.md            # Project description, setup, and usage instructions
├── docker-compose.yml   # Defines multi-container Docker applications (e.g., Dagster services)
├── logs.log             # A specific log file (content not inspected)
├── pyproject.toml       # Project metadata and dependencies (PEP 517/518/621)
└── uv.lock              # Pinned dependency versions managed by uv
```

## 2. Source Code (`src/clustering/`)

The core application logic resides within the `src/clustering` Python package. It is further organized into submodules:

*   **`__init__.py`**: Package initializer.
*   **`__main__.py`**: Allows the package to be run as a script (e.g., `python -m clustering`).
*   **`cli/`**: Command-line interface implementation (content not inspected).
*   **`core/`**: Likely contains the main business logic, algorithms (e.g., clustering algorithms), and data structures (content not inspected).
*   **`dagster/`**: Contains all Dagster-specific definitions (assets, resources, jobs). See Section 3.
*   **`infra/`**: Infrastructure-related code, such as configuration loading (`hydra_config.py` was referenced).
*   **`io/`**: Input/Output operations, potentially data loading/saving utilities independent of Dagster resources (content not inspected).
*   **`utils/`**: General utility functions shared across the package (content not inspected).
*   **`README.md`**: Specific documentation for the `clustering` package.
*   **`py.typed`**: Marker file indicating the package provides type hints (PEP 561).

## 3. Dagster Implementation (`src/clustering/dagster/`)

This directory orchestrates the data pipeline using Dagster.

*   **`__init__.py`**: Sub-package initializer.
*   **`app.py`**: Likely contains code to serve the Dagster UI/API, possibly using `dagster-webserver`.
*   **`definitions.py`**: The central file defining the Dagster repository. It includes:
    *   **Asset Imports**: Imports numerous asset functions from `clustering.dagster.assets`. Assets are categorized into "internal" and "external" tracks, covering stages like preprocessing, feature engineering, model training, cluster assignment, and analysis. A final "merging" stage is also present.
    *   **Asset Grouping**: Assets are grouped into Python lists (e.g., `internal_preprocessing_assets`) for easier management in job definitions.
    *   **Job Definitions**: Defines several `dg.define_asset_job` instances:
        *   `internal_preprocessing_job`
        *   `external_preprocessing_job`
        *   `internal_ml_job` (combines feature engineering, training, assignment, analysis)
        *   `external_ml_job` (combines feature engineering, training, assignment, analysis)
        *   `merging_job`
        *   `full_pipeline_job` (selects all assets, configured for sequential execution).
    *   **Configuration Loading**: Implements `load_config` function to load environment-specific (`dev`, `staging`, `prod`) configurations from YAML files using a Hydra-like resolver (`clustering.infra.hydra_config.load_config`).
    *   **Resource Definition**: Implements `get_resources_by_env` to dynamically create resource definitions based on the environment and loaded configuration. Key resources include:
        *   `io_manager`: `dg.FilesystemIOManager` (stores data in `storage/`).
        *   `job_params`/`config`: Provides configuration values via a `SimpleNamespace`.
        *   `logger`: Configured `logger_service`.
        *   Multiple configured `data_reader` and `data_writer` instances from `clustering.dagster.resources.data_io` for various data sources/sinks.
    *   **Definitions Creation**: The `create_definitions` function assembles assets, resources, and jobs into a `dg.Definitions` object. A default instance (`defs`) is created for the "dev" environment.
*   **`assets/`**: Contains the implementations of the Software-Defined Assets (SDAs).
    *   `__init__.py`: Imports and potentially groups assets from subdirectories.
    *   `preprocessing/`: Assets related to data preprocessing.
    *   `merging/`: Assets related to merging results from internal/external pipelines.
    *   `clustering/`: Assets related to the core clustering steps (training, assignment, analysis).
*   **`resources/`**: Contains implementations for Dagster resources.
    *   `__init__.py`: Sub-package initializer.
    *   `logging.py`: Implementation for the `logger_service`.
    *   `data_io.py`: Implementation for the `data_reader` and `data_writer` configurable resources.
    *   `configs/`: Contains environment-specific configuration files (e.g., `dev.yml`, `staging.yml`, `prod.yml`). These files define parameters for jobs, logging, and data I/O resources.
*   **`README.md`**: Specific documentation for the Dagster implementation.

## 4. Configuration

*   **`pyproject.toml`**: Defines project metadata, dependencies (likely includes `dagster`, `pandas`, ML libraries), and tool configurations (e.g., `ruff`, `uv`).
*   **`uv.lock`**: Pins exact versions of all dependencies for reproducible environments. Managed by `uv`.
*   **`.env` / `.env.example`**: Store environment variables (e.g., API keys, file paths). Variables seem to be resolved into the Dagster configuration via the Hydra loader.
*   **`src/clustering/dagster/resources/configs/*.yml`**: Environment-specific configurations for Dagster resources and job parameters.

## 5. Testing (`tests/`)

Contains automated tests. The specific frameworks (`pytest`, `unittest`) and coverage are not determined from the current analysis but its presence indicates adherence to testing practices.

## 6. Containerization

*   **`Dockerfile`**: Defines how to build a Docker image containing the application and its dependencies. Likely used for deployment or ensuring a consistent runtime environment.
*   **`docker-compose.yml`**: Used to define and run multi-container Docker applications. Likely sets up the Dagster webserver, daemon, and potentially other services (e.g., databases, message queues if used).
*   **`.devcontainer/`**: Configuration for using VS Code Dev Containers, ensuring a consistent development environment for all contributors.

## 7. Key Files

*   **`Makefile`**: Provides convenient command shortcuts (`make lint`, `make test`, `make run`) for development workflows.
*   **`README.md`**: Top-level project documentation.
*   **`.gitignore`**: Standard Git ignore file.
*   **`.pre-commit-config.yaml`**: Configures pre-commit hooks (e.g., linters, formatters) to ensure code quality before commits.

This overview provides a snapshot based on the file structure and the content of `src/clustering/dagster/definitions.py`. Deeper insights would require inspecting the code within assets, core logic, tests, and configuration files.
