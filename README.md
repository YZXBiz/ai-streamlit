# Store Clustering

A Python package for internal and external store clustering analysis.

## Overview

This project provides tools for clustering retail stores based on various metrics and characteristics. It includes:

- Internal clustering based on store performance metrics
- External clustering based on competitive and market data
- Tools for merging internal and external clustering results
- Preprocessing pipelines for data preparation
- Dagster orchestration for workflow management
- MLflow for experiment tracking and model management

## Installation

### Prerequisites

- Python 3.11 or higher
- uv (Python package manager)
- Docker and Docker Compose (optional, for containerization)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/cvshealth/store-clustering.git
   cd store-clustering
   ```

2. Create your `.env` file:

   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. Install dependencies:

   ```bash
   make install  # For production
   # OR
   uv venv  # Create virtual environment
   uv sync --all-packages  # Install all packages
   ```

4. For development, install additional dependencies:

   ```bash
   uv pip install --python-file pyproject.toml -e ".[dev]"
   ```

5. Set up configuration:
   ```bash
   make setup-configs
   ```

### VS Code/Cursor Extensions

This project includes recommended VS Code/Cursor extensions to improve the development experience:

#### For VS Code users:
- When you open the project, you'll see a notification to install recommended extensions
- Click "Install All" to install them automatically

#### For Cursor users:
1. Open the Extensions tab (or press Ctrl+Shift+X / Cmd+Shift+X)
2. Type `@recommended` in the search bar
3. Click "Install All" to install all recommended extensions

The recommended extensions include Python, Dagster, linting, and formatting tools that help maintain code quality and consistency.

## Project Structure

```
store-clustering/
├── configs/              # Configuration files for jobs
├── context/              # Project documentation and context
├── src/                  # Source code
│   └── clustering/       # Main package
│       ├── core/         # Core models and schemas
│       ├── io/           # Input/output utilities
│       ├── utils/        # Utility functions
│       ├── infra/        # Infrastructure components
│       ├── cli/          # Command-line interface tools
│       └── dagster/      # Dagster workflow definitions
│           ├── assets/   # Dagster assets
│           ├── resources/# Dagster resources
│           ├── sensors/  # Dagster sensors for triggering workflows
│           ├── schedules/# Dagster schedules
│           └── partitions/# Dagster partitions
├── tests/                # Test suite
│   ├── integration/      # Integration tests
│   └── unit/             # Unit tests
├── outputs/              # Job outputs
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── Makefile              # Development and build commands
├── pyproject.toml        # Project metadata and dependencies
├── Dockerfile            # Docker container definition
└── docker-compose.yml    # Docker Compose configuration
```

## Usage

### Running Jobs

#### Using Make Commands

```bash
# Run a specific job (legacy method)
make run-internal_preprocessing
make run-internal_clustering
make run-external_preprocessing
make run-external_clustering

# Run all jobs
make run-all
```

#### Using Dagster

```bash
# Start the Dagster UI
make dagster-ui

# Run a specific Dagster job
make run-dagster-internal_preprocessing
make run-dagster-internal_clustering

# Alternative: Run using the Dagster CLI
make dagster-job-internal_preprocessing
```

#### Running Directly with uv

```bash
# Run a specific job
uv run -m clustering configs/internal_preprocessing.yml

# Run a Dagster job
uv run -m clustering.dagster.run_dagster internal_preprocessing --env dev
```

#### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Run a specific job
docker-compose run clustering configs/internal_preprocessing.yml
```

### Development

```bash
# Format code
make format
# Or directly:
uv run -m ruff format src tests

# Run linting
make lint
# Or directly:
uv run -m ruff check src tests --fix

# Run type checking
make type-check
# Or directly:
uv run -m mypy src tests

# Run tests
make test
# Or directly:
uv run -m pytest tests --cov=src

# Clean up artifacts
make clean
```

## MLflow Integration

The project includes MLflow for experiment tracking and model management. When running the application, MLflow server can be started with:

```bash
# Using Docker Compose
docker-compose up -d mlflow

# Or directly
uv run -m mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns
```

Then access the MLflow UI at http://localhost:5000.

## Configuration

Create YAML configuration files in the `configs/` directory for each job:

- `internal_preprocessing.yml`
- `internal_clustering.yml`
- `external_preprocessing.yml`
- `external_clustering.yml`

## Contributing

1. Ensure you have setup the development environment
2. Create a new branch for your feature
3. Add tests for your changes
4. Run tests to ensure they pass
5. Submit a pull request

## Contact

For questions or support regarding this project, please contact:

**Jackson Yang**  
Email: Jackson.Yang@cvshealth.com

## License

[Specify license information]
