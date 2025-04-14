# Store Clustering Data Pipeline

![Dagster](https://img.shields.io/badge/orchestration-Dagster-green)
![Python](https://img.shields.io/badge/language-Python_3.10-blue)
![License](https://img.shields.io/badge/license-Proprietary-red)

A data pipeline for clustering stores based on sales data and external data sources, built with Dagster.

## 🎯 Project Purpose

This project implements a comprehensive data processing and clustering pipeline for store analysis. It processes both internal sales data and external data sources, applies feature engineering, trains clustering models, and provides tools for analyzing the resulting clusters.

The primary goal is to identify meaningful store segments that can inform business strategy, merchandising decisions, and marketing initiatives.

## 📋 Table of Contents

- [Store Clustering Data Pipeline](#store-clustering-data-pipeline)
  - [Project Purpose](#-project-purpose)
  - [Table of Contents](#-table-of-contents)
  - [Features](#-features)
  - [Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Pipeline](#running-the-pipeline)
  - [Architecture](#-architecture)
    - [Pipeline Structure](#pipeline-structure)
    - [Data Flow](#data-flow)
  - [Configuration](#-configuration)
  - [Development](#-development)
    - [Code Quality](#code-quality)
    - [Testing](#testing)
  - [Documentation](#-documentation)
  - [FAQ](#-faq)
  - [Project Ownership](#-project-ownership)
  - [License](#-license)

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

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://astral.sh/uv) package manager

### Installation

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

## ⚙️ Configuration

Configuration is managed through YAML files located in `src/clustering/dagster/resources/configs/`:

- `dev.yml`: Development environment config
- `staging.yml`: Staging environment config
- `prod.yml`: Production environment config

Key configuration parameters include:

- Feature engineering settings (normalization, imputation, outlier detection)
- Model training parameters (algorithm, min/max clusters)
- Data source and destination paths
- Logging configuration

To use a specific environment:

```bash
make full-pipeline ENV=prod
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

## ❓ FAQ

**Q: How do I determine the optimal number of clusters?**  
A: The pipeline automatically evaluates different cluster counts based on silhouette scores, Calinski-Harabasz Index, and Davies-Bouldin Index. You can configure the range with `min_clusters` and `max_clusters` in the config file.

**Q: Can I run the pipeline with limited memory?**  
A: Yes, use `make run-memory-optimized JOB=job_name` to run with memory optimization enabled.

**Q: How do I add a new data source?**  
A: Add a new reader configuration in the environment config file and create a corresponding asset in the appropriate preprocessing module.

## 👥 Project Ownership

**Author**: Jackson Yang  
**Email**: Jackson.Yang@cvshealth.com  
**Organization**: CVS Health

## 📄 License

Copyright © 2025 CVS Health. All rights reserved.
