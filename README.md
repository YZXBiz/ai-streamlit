# Store Clustering

A Python package for internal and external store clustering analysis.

## Overview

This project provides tools for clustering retail stores based on various metrics and characteristics. It includes:

- Internal clustering based on store performance metrics
- External clustering based on competitive and market data
- Tools for merging internal and external clustering results
- Preprocessing pipelines for data preparation

## Installation

### Prerequisites

- Python 3.11 or higher
- uv (Python package manager)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/store-clustering.git
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
   make dev-install  # For development
   ```

4. Set up configuration:
   ```bash
   make setup-configs
   ```

## Project Structure

```
store-clustering/
├── configs/              # Configuration files for jobs
├── context/              # Project documentation and context
├── src/                  # Source code
│   └── clustering/       # Main package
│       ├── core/         # Core models and schemas
│       ├── io/           # Input/output utilities
│       ├── jobs/         # Job implementations
│       └── utils/        # Utility functions
├── tests/                # Test suite
│   ├── integration/      # Integration tests
│   └── unit/             # Unit tests
├── .env.example          # Example environment variables
├── .gitignore            # Git ignore file
├── Makefile              # Development and build commands
└── pyproject.toml        # Project metadata and dependencies
```

## Usage

### Running Jobs

Use the provided Makefile targets to run jobs:

```bash
# Run a specific job
make run-internal_preprocessing
make run-internal_clustering
make run-external_preprocessing
make run-external_clustering

# Run the entire pipeline
make run-all
```

### Development

```bash
# Format code
make format

# Run linting
make lint

# Run type checking
make type-check

# Run tests
make test

# Clean up artifacts
make clean
```

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

## License

[Specify license information] 