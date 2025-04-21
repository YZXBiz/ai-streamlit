# Store Clustering Data Pipeline

![Dagster](https://img.shields.io/badge/orchestration-Dagster-green)
![Python](https://img.shields.io/badge/language-Python_3.10-blue)
[![Coverage Status](https://coveralls.io/repos/github/YOUR_USERNAME/clustering-dagster/badge.svg?branch=main)](https://coveralls.io/github/YOUR_USERNAME/clustering-dagster?branch=main)

A data pipeline for clustering stores based on sales data and external data sources, built with Dagster.

## 📋 Table of Contents

- [Store Clustering Data Pipeline](#store-clustering-data-pipeline)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 Project Purpose](#-project-purpose)
  - [✨ Features](#-features)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
    - [Environment Setup](#environment-setup)
    - [Running the Pipeline](#running-the-pipeline)
    - [Basic Usage](#basic-usage)
    - [Troubleshooting](#troubleshooting)
  - [⚙️ Configuration](#️-configuration)
    - [Data Directory Structure](#data-directory-structure)
    - [Configuration Files](#configuration-files)
  - [📊 Usage](#-usage)
    - [Using the CLI](#using-the-cli)
      - [Getting Help](#getting-help)
    - [Running the Pipeline](#running-the-pipeline-1)
    - [Common Workflows](#common-workflows)
    - [Error Handling](#error-handling)
  - [🏗️ Architecture](#️-architecture)
    - [Pipeline Structure](#pipeline-structure)
    - [Data Flow](#data-flow)
  - [💻 Development](#-development)
    - [Code Quality](#code-quality)
    - [Testing](#testing)
      - [Code Coverage](#code-coverage)
  - [📚 Documentation](#-documentation)
  - [🔒 Security \& Privacy](#-security--privacy)
    - [Package Verification](#package-verification)
    - [Data Collection](#data-collection)
  - [❓ FAQ](#-faq)
  - [🚀 Deployment](#-deployment)
    - [GitLab CI/CD Setup](#gitlab-cicd-setup)
      - [Pipeline Stages](#pipeline-stages)
      - [Deployment Environments](#deployment-environments)
    - [Manual Deployment](#manual-deployment)
  - [How to Run](#how-to-run)
    - [1. Dagster Web UI (Recommended)](#1-dagster-web-ui-recommended)
    - [2. Command Line Interface](#2-command-line-interface)
    - [3. Visualization Dashboard](#3-visualization-dashboard)
    - [4. Scheduled Execution](#4-scheduled-execution)

## 🎯 Project Purpose

This project implements a comprehensive data processing and clustering pipeline for store analysis. It processes both internal sales data and external data sources, applies feature engineering, trains clustering models, and provides tools for analyzing the resulting clusters.

The primary goal is to identify meaningful store segments that can inform business strategy, merchandising decisions, and marketing initiatives.

## ✨ Features

- **Modular Pipeline Design**: Separate jobs for internal data, external data, and cluster merging
- **Flexible Configuration**: Environment-specific config files (dev, staging, prod)
- **Advanced Feature Engineering**:
  - Normalization (robust scaling)
  - Missing value imputation
  - Outlier detection and removal
  - Dimensionality reduction (PCA)
- **Optimal Cluster Selection**: Automated determination of optimal cluster counts
- **Model Training**: KMeans clustering with configurable parameters
- **Cluster Analysis**: Metrics calculation and visualization generation
- **Memory Optimization**: Support for memory-optimized processing of large datasets

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- [UV package manager](https://docs.astral.sh/uv/) (recommended)
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/clustering-dagster.git
   cd clustering-dagster
   ```

2. Install the package:
   ```bash
   # Standard installation
   uv add .
   
   # Or install in development mode
   uv add -e ".[dev]"  # Include development tools
   ```

3. Create required directories:
   ```bash
   # Create required data directories
   mkdir -p data/internal data/external data/merging data/raw

   # Create the Dagster home directory
   mkdir -p dagster_home
   ```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Create a .env file from the template
cp .env.example .env
```

Edit the `.env` file to configure your environment variables:

```
# Base directories
DATA_DIR=./data
INTERNAL_DATA_DIR=./data/internal
EXTERNAL_DATA_DIR=./data/external
MERGING_DATA_DIR=./data/merging

# Dagster configuration
DAGSTER_HOME=./dagster_home
```

### Running the Pipeline

Verify installation:

```bash
# Verify the CLI functionality
clustering --version
```

Start the Dagster development server:

```bash
# Start the Dagster development server
make dev
```

Then open your browser at [http://localhost:3000](http://localhost:3000) to access the Dagster UI.

### Basic Usage

Using the CLI:

```bash
# Run the CLI with help command to see available options
clustering --help

# Execute specific CLI commands
clustering validate --config your_config.yaml
```

Running the Dashboard:

```bash
# Start the dashboard
clustering dashboard
```

### Troubleshooting

If you encounter issues:

1. Ensure all required directories exist
2. Verify your environment variables in the `.env` file
3. Check that your Python version is 3.10 or higher
4. Make sure Dagster can access the required data directories
5. Check logs in the `dagster_home/logs` directory

## ⚙️ Configuration

### Data Directory Structure

Create the required data directory structure:

```bash
mkdir -p data/internal data/external data/merging data/raw
```

The project expects the following data directory structure:

```
data/
├── internal/       # Internal sales and product data
├── external/       # External data sources
├── merging/        # Output from cluster merging process
└── raw/            # Raw data files before processing
```

Each directory contains intermediate files produced by the pipeline, such as:
- Processed sales data
- Feature engineered datasets
- Trained models
- Cluster assignments

### Configuration Files

1. **YAML Configuration Files**:
   The pipeline requires specific YAML configuration files:

   ```bash
   mkdir -p configs/
   ```

   Create the following configuration files:

   ```bash
   # dev.yml example - copy this into configs/dev.yml
   cat << EOF > configs/dev.yml
   paths:
     base_data_dir: \${env:DATA_DIR,./data}
     internal_data_dir: \${env:INTERNAL_DATA_DIR,./data/internal}
     external_data_dir: \${env:EXTERNAL_DATA_DIR,./data/external}
     merging_data_dir: \${env:MERGING_DATA_DIR,./data/merging}

   preprocessing:
     normalize: true
     impute_missing: true
     outlier_removal: true

   model:
     algorithm: kmeans
     min_clusters: 3
     max_clusters: 10
   EOF
   ```

2. **Configuration Parameters**:
   Key configuration parameters include:

   - Feature engineering settings (normalization, imputation, outlier detection)
   - Model training parameters (algorithm, min/max clusters)
   - Data source and destination paths
   - Logging configuration

## 📊 Usage

### Using the CLI

The project includes a command-line interface for interacting with the Dagster-based clustering pipeline.

#### Getting Help

View all available commands:

```bash
clustering --help

# View help for a specific command
clustering run --help
```

Example output:
```
Usage: clustering [OPTIONS] COMMAND [ARGS]...

  Clustering Pipeline CLI - Process and analyze store data

Options:
  --version  Show version information
  --help     Show this message and exit

Commands:
  run         Run a specific pipeline job
  ui          Launch the Dagster web UI
  list-jobs   List available pipeline jobs
  minimal     Run a minimal demo
```

### Running the Pipeline

The project includes several run configurations:

1. **Development Server**:
   ```bash
   clustering ui
   ```
   This launches the Dagster UI at http://localhost:3000

2. **Full Pipeline**:
   ```bash
   clustering run full_pipeline_job
   ```
   This runs the complete pipeline including internal preprocessing, model training, external data integration, and cluster merging.

3. **Individual Pipeline Components**:
   ```bash
   clustering run internal_preprocessing_job  # Run internal data preprocessing
   clustering run internal_ml_job             # Run internal ML pipeline
   clustering run external_preprocessing_job  # Run external data preprocessing
   clustering run external_ml_job             # Run external ML pipeline
   clustering run merging_job                 # Run cluster merging
   ```

4. **Memory-Optimized Mode**:
   ```bash
   clustering run full_pipeline_job --memory-optimized
   ```

### Common Workflows

Here are some common task examples combining multiple commands:

```bash
# 1. Full development workflow
clustering list-jobs && \
clustering run internal_preprocessing_job --env dev && \
clustering ui

# 2. Production deployment with monitoring
clustering run full_pipeline_job --env prod --tags version=2.0 && \
clustering ui --host 0.0.0.0 --port 8080

# 3. Quick test with minimal setup
clustering minimal
```

### Error Handling

The CLI provides detailed error messages and exit codes:

- Exit code 0: Success
- Exit code 1: General error
- Exit code 2: Invalid arguments
- Exit code 3: Environment configuration error
- Exit code 4: Pipeline execution error

Common error scenarios and solutions:

1. **Configuration Errors**:
   ```
   Error: No configuration found for environment 'prod'
   Solution: Ensure .env.prod exists or use --env dev
   ```

2. **Pipeline Failures**:
   ```
   Error: Job 'internal_preprocessing_job' failed
   Solution: Check logs at ~/.clustering/logs/
   ```

3. **Permission Issues**:
   ```
   Error: Unable to access data directory
   Solution: Verify permissions and paths in config
   ```

## 🏗️ Architecture

### Pipeline Structure

The pipeline consists of several distinct Dagster jobs:

1. **Internal Preprocessing**: Processes raw sales data and product category mappings
2. **External Preprocessing**: Processes external data sources
3. **Internal ML**: Applies feature engineering and clustering to internal data
4. **External ML**: Applies feature engineering and clustering to external data
5. **Merging**: Combines internal and external clusters into unified store segments
6. **Full Pipeline**: Orchestrates all the above jobs in sequence

### Data Flow

```
Raw Sales Data → Normalization → Feature Engineering → Model Training → Cluster Assignment
     ↑                                                       ↑
External Data → Preprocessing → Feature Engineering → Model Training
     ↓                                                       ↓
                        Merged Cluster Analysis & Assignment
```

## 💻 Development

### Code Quality

Maintain code quality with the following tools configured in the project:

```bash
# Format code with ruff
make format

# Lint code and auto-fix issues
make lint

# Run type checking
make type-check
```

### Testing

Tests are organized by component:

```bash
# Run all tests
make test

# Run specific test categories
make test-unit        # Run all unit tests
make test-integration # Run integration tests
make test-cli         # Run CLI package tests
make test-shared      # Run shared package tests
make test-pipeline    # Run pipeline tests
```

#### Code Coverage

Generate coverage reports:

```bash
# Generate coverage reports
make test-coverage

# Generate HTML coverage report
make test-html-coverage
```

## 📚 Documentation

Build and view documentation:

```bash
# Build documentation
make docs

# Start documentation server
make docs-serve
```

Open `docs/build/html/index.html` in your browser to view.

## 🔒 Security & Privacy

### Package Verification

All dependencies are managed using the UV package manager:

```bash
# Package management - always use UV, never pip
uv add <package-name>
```

### Data Collection

This project does not collect any telemetry or usage data by default. All data processing happens locally within your infrastructure.

## ❓ FAQ

**Q: How do I determine the optimal number of clusters?**
A: The pipeline automatically evaluates different cluster counts based on silhouette scores, Calinski-Harabasz Index, and Davies-Bouldin Index. You can configure the range with `min_clusters` and `max_clusters` in the config file.

**Q: Can I run the pipeline with limited memory?**
A: Yes, use the `--memory-optimized` flag when running jobs to enable memory optimization.

**Q: How do I add a new data source?**
A: Add a new reader configuration in the environment config file and create a corresponding asset in the appropriate preprocessing module.

**Q: Where can I find the project's dependencies?**
A: All dependencies are listed in the `pyproject.toml` file, categorized into core dependencies and optional dependencies.

**Q: How do I create a new Dagster asset?**
A: Follow the asset creation guide. Assets should be organized by their purpose in appropriate directories (preprocessing, clustering, checks, merging), use the `@asset` decorator, and have explicit dependencies declared.

**Q: How should I handle store numbers in the data?**
A: Always ensure `STORE_NBR` is handled as a string, not an integer.

**Q: What type should cluster counts be?**
A: Ensure optimal cluster counts are integers, not floats.

**Q: How do I validate my DataFrame before processing?**
A: Always validate dataframe schemas before processing using the schema validation utilities.

## 🚀 Deployment

The store clustering pipeline supports deployment with GitLab CI/CD for automated building, testing, and deployment to multiple environments.

### GitLab CI/CD Setup

The project includes a complete GitLab CI/CD pipeline for deploying the Dagster pipeline to development, staging, and production environments. See [Deployment Guide](docs/deployment.md) for detailed instructions.

#### Pipeline Stages

1. **Validate**: Ensures pipeline definitions are valid
2. **Test**: Runs unit and integration tests
3. **Build**: Builds Docker image for deployment
4. **Test Deployment**: Validates environment configuration
5. **Deploy**: Deploys the pipeline to target environment

#### Deployment Environments

- **Development**: Automatically validated for all branches
- **Staging**: Deployed manually from main branch
- **Production**: Deployed manually from version tags

### Manual Deployment

For manual deployment without CI/CD:

```bash
# Build Docker image
docker build -t dagster-pipeline:latest .

# Run with docker-compose
docker-compose up -d
```

## How to Run

After installation, there are several ways to run the clustering pipeline:

### 1. Dagster Web UI (Recommended)

Start the Dagster web interface:

```bash
# Using make
make dev

# Or directly
uv run -m dagster dev -m clustering.pipeline.definitions
```

Then open `http://localhost:3000` in your browser to:
- View all assets and their dependencies
- Run jobs with custom configurations
- Monitor execution progress
- Explore asset materializations

### 2. Command Line Interface

Run specific pipeline jobs:

```bash
# Run full pipeline with config file
uv run -m clustering.cli.commands run full_pipeline --config path/to/config.yaml

# Run with inline parameters
uv run -m clustering.cli.commands run cluster_stores --param cluster_count=5
```

### 3. Visualization Dashboard

Launch the analytics dashboard:

```bash
# Using make
make dashboard

# Or directly
uv run -m clustering.dashboard
```

### 4. Scheduled Execution

Set up scheduled runs in the Dagster UI or configure schedules programmatically.

For production deployment, consider:
- Using Dagster Daemon for scheduled runs
- Setting up proper environment variables
- Configuring appropriate storage solutions

For detailed configuration options, see the [Installation Guide](INSTALLATION_GUIDE.md).

