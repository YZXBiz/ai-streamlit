# Clustering Pipeline

A modular, scalable clustering pipeline built with Dagster for processing and analyzing internal and external datasets.

## Overview

This package implements a comprehensive clustering pipeline that:

- Processes both internal sales data and external datasets
- Performs feature engineering with configurable preprocessing steps
- Trains clustering models with optimal cluster count detection
- Assigns data points to clusters and generates visualizations
- Merges results from different clustering approaches

## Project Structure

```
clustering/
├── core/                  # Domain models and schemas
├── dagster/               # Dagster pipeline components
│   ├── assets/            # All asset definitions
│   ├── resources/         # Resource definitions
│   └── definitions.py     # Main pipeline definitions
├── io/                    # I/O handling components
├── utils/                 # Utility functions
└── cli/                   # Command-line interface
```

## Dagster Pipeline

The pipeline is organized into several key jobs:

1. **Internal Preprocessing**: Transforms raw internal sales data
2. **External Preprocessing**: Prepares external feature data
3. **Internal ML**: Runs clustering on internal data
4. **External ML**: Runs clustering on external data
5. **Merging**: Combines the internal and external clustering results
6. **Full Pipeline**: End-to-end execution of all steps

## Usage

### Running with CLI

```bash
# Start the Dagster UI
python -m clustering ui

# Run a specific job
python -m clustering run internal_preprocessing_job --env dev

# Run the full pipeline
python -m clustering run full_pipeline_job --env prod
```

### Using Dagster UI

Access the Dagster UI at http://localhost:3000 to:
- Monitor and manage jobs
- View asset materializations
- Inspect job execution graphs
- Launch ad-hoc runs

## Configuration

Configuration is environment-based and loaded from YAML files:

```
dagster/resources/configs/
├── dev.yml
├── staging.yml
└── prod.yml
```

Each file configures:
- Job parameters
- Logging configuration
- Data reader/writer settings

## Assets

The pipeline is built around these main asset categories:

### Preprocessing Assets
- Raw data loading
- Category mapping
- Normalization

### Feature Engineering Assets
- Filtering
- Imputation
- Normalization
- Dimensionality reduction

### Model Training Assets
- Optimal cluster count detection
- Model training
- Model persistence

### Analysis Assets
- Cluster assignment
- Metric calculation
- Visualization generation

### Merging Assets
- Combined cluster mapping
- Small cluster reassignment
- Final cluster persistence

## Development

### Prerequisites

- Python 3.10+
- Dependencies in requirements.txt

### Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit
```

## License

[License information]
