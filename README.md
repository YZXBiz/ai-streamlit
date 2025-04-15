# Store Clustering Data Pipeline

![Dagster](https://img.shields.io/badge/orchestration-Dagster-green)
![Python](https://img.shields.io/badge/language-Python_3.10-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)

A data pipeline for clustering stores based on sales data and external data sources, built with Dagster.

## 📋 Table of Contents

- [Store Clustering Data Pipeline](#store-clustering-data-pipeline)
  - [Project Information](#-project-information)
  - [Project Purpose](#-project-purpose)
  - [Features](#-features)
  - [Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [Developer Installation](#developer-installation)
    - [Installing as a Package](#installing-as-a-package)
    - [Uninstallation](#uninstallation)
    - [Shell Completion](#shell-completion)
  - [Configuration](#-configuration)
    - [Environment Setup](#environment-setup)
    - [Data Directory Structure](#data-directory-structure)
    - [Configuration Files](#configuration-files)
  - [Usage](#-usage)
    - [Using the CLI](#using-the-cli)
    - [Running the Pipeline](#running-the-pipeline)
    - [Common Workflows](#common-workflows)
    - [Error Handling](#error-handling)
  - [Architecture](#-architecture)
    - [Pipeline Structure](#pipeline-structure)
    - [Data Flow](#data-flow)
  - [Development](#-development)
    - [Code Quality](#code-quality)
    - [Testing](#testing)
  - [Documentation](#-documentation)
  - [Security & Privacy](#-security--privacy)
  - [FAQ](#-faq)

## 👥 Project Information

**Author**: Jackson Yang  
**Email**: Jackson.Yang@cvshealth.com  
**Organization**: CVS Health
**License**: Copyright © 2025 CVS Health. All rights reserved.

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

- Python 3.10+
- [uv](https://astral.sh/uv) package manager

### Developer Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/testing-dagster.git
   cd testing-dagster
   ```

2. Install dependencies:
   ```bash
   make install
   ```

3. Verify installation:
   ```bash
   make version
   ```

### Installing as a Package

To install this project as a Python package:

```bash
# Install specific version (recommended)
uv add clustering-pipeline==1.0.0

# Install from the local directory
uv add -e .

# Or install directly from GitHub
uv add git+https://github.com/yourusername/testing-dagster.git@v1.0.0
```

### Uninstallation

To remove the package:

```bash
# Remove the package
uv remove clustering-pipeline

# Remove all related configuration (optional)
rm -rf ~/.clustering
```

### Shell Completion

Enable shell completion for easier CLI usage:

```bash
# For Bash
clustering completion bash > ~/.bash_completion.d/clustering

# For Zsh
clustering completion zsh > "${fpath[1]}/_clustering"

# For Fish
clustering completion fish > ~/.config/fish/completions/clustering.fish
```

## ⚙️ Configuration

### Environment Setup

1. **Environment Variables**:
   The project includes an example environment file (`.env.example`) that you should copy and customize:

   ```bash
   # Copy the example environment file for development
   cp .env.example .env.dev
   
   # For production (optional)
   cp .env.example .env.prod
   
   # Edit the files with your specific configuration values
   # Example .env file structure:
   # DATA_DIR=/path/to/your/data
   # INTERNAL_DATA_DIR=/path/to/your/data/internal
   # EXTERNAL_DATA_DIR=/path/to/your/data/external
   # MERGING_DATA_DIR=/path/to/your/data/merging
   ```

   > **Note:** `.env` files are included in `.gitignore` and should never be committed to version control as they may contain sensitive information.

2. **Automated Setup** (Optional):
   For convenience, you can run the included setup script which creates all necessary configuration and directories:

   ```bash
   # Run the setup script
   ./scripts/setup.sh
   
   # Or on Windows
   .\scripts\setup.bat
   ```

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

   To use a specific environment:

   ```bash
   make full-pipeline ENV=prod
   ```

## 📊 Usage

### Using the CLI

The project includes a command-line interface for interacting with the Dagster-based clustering pipeline.

#### Getting Help

View all available commands:

```bash
clustering --help

# View help for a specific command
clustering run --help

# Access man pages (if installed)
man clustering
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
   make dev
   ```
   This launches the Dagster UI at http://localhost:3000

2. **Full Pipeline**:
   ```bash
   make run-full
   ```
   This runs the complete pipeline including internal preprocessing, model training, external data integration, and cluster merging.

3. **Individual Pipeline Components**:
   ```bash
   make run-internal-preprocessing  # Run internal data preprocessing
   make run-internal-ml             # Run internal ML pipeline
   make run-external-preprocessing  # Run external data preprocessing
   make run-external-ml             # Run external ML pipeline
   make run-merging                 # Run cluster merging
   ```

4. **Memory-Optimized Mode**:
   ```bash
   make run-memory-optimized JOB=full_pipeline_job
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
   ```bash
   Error: No configuration found for environment 'prod'
   Solution: Ensure .env.prod exists or use --env dev
   ```

2. **Pipeline Failures**:
   ```bash
   Error: Job 'internal_preprocessing_job' failed
   Solution: Check logs at ~/.clustering/logs/
   ```

3. **Permission Issues**:
   ```bash
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

Maintain code quality with the provided tools:

```bash
make format         # Format code with ruff
make lint           # Lint code and auto-fix issues
make type-check     # Run type checking with mypy and pyright
make check-all      # Run all code quality checks
```

### Testing

Run tests with:

```bash
make test           # Run all tests with coverage
make dagster-test   # Run Dagster-specific tests
```

## 📚 Documentation

Build and view documentation with:

```bash
make docs           # Build documentation
```

Open `docs/build/html/index.html` in your browser to view.

Or start the documentation server:

```bash
make docs-server    # Start documentation server at http://localhost:8000
```

## 🔒 Security & Privacy

### Package Verification

Verify package integrity during installation:

```bash
# Download and verify package signature
uv add clustering-pipeline --require-hashes

# View package metadata
uv pip show clustering-pipeline
```

### Data Collection

This CLI does not collect any telemetry or usage data by default. All data processing happens locally within your infrastructure.

## ❓ FAQ

**Q: How do I determine the optimal number of clusters?**  
A: The pipeline automatically evaluates different cluster counts based on silhouette scores, Calinski-Harabasz Index, and Davies-Bouldin Index. You can configure the range with `min_clusters` and `max_clusters` in the config file.

**Q: Can I run the pipeline with limited memory?**  
A: Yes, use `make run-memory-optimized JOB=job_name` to run with memory optimization enabled.

**Q: How do I add a new data source?**  
A: Add a new reader configuration in the environment config file and create a corresponding asset in the appropriate preprocessing module.
