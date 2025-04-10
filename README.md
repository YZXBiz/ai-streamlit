# Store Clustering

An advanced Python package for retail store clustering analysis using Dagster for workflow orchestration.

![Store Clustering](https://img.shields.io/badge/Store-Clustering-blue)
![Dagster](https://img.shields.io/badge/Dagster-v1.10.4-orange)
![Python](https://img.shields.io/badge/Python-v3.11+-green)

## 🌟 Highlights

- **Dual Clustering Pipeline**: Internal (store performance) and external (market data) clustering with automatic merging
- **Dagster Asset-based Architecture**: Modular, reproducible asset-based data pipeline with comprehensive versioning
- **Advanced Feature Engineering**: Automated feature selection, normalization, imputation, and outlier removal
- **Configurable Workflows**: YAML-based configuration with environment-specific settings
- **Intelligent Cluster Optimization**: Automatic small cluster reassignment to nearest large clusters

## 📋 Overview

This project provides a comprehensive solution for clustering retail stores based on multiple data sources. The pipeline processes both internal performance metrics and external market data to create refined clusters, then intelligently merges them for a holistic view of store characteristics.

The project uses Dagster's asset-based paradigm for workflow orchestration, allowing for:
- Granular asset versioning and dependency tracking
- Intelligent materialization based on code and data version changes
- Automated data quality checks and monitoring
- Intuitive visualizations of the pipeline and results

### 🔄 Data Versioning

The project supports robust data versioning through Dagster's built-in versioning system:

- **Asset Versioning**: Each asset has a `code_version` that tracks when code logic changes
- **Data Versioning**: Inputs and outputs are automatically tracked with data versions
- **Caching**: Unnecessary recomputations are avoided when code or data hasn't changed
- **DVC Integration**: For versioning large data files outside of code repositories

### 🏗️ Architecture

The pipeline is structured into three main stages:

1. **Internal Clustering**: Processes internal store metrics through preprocessing, feature engineering, model training, and cluster assignment
2. **External Clustering**: Similar workflow for external market and competitive data
3. **Merging**: Combines internal and external clusters, with optimization to reassign small clusters

## 🛠️ Installation

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

## 📊 Usage

### Running the Pipeline

#### Using Make Commands

```bash
# Start the Dagster UI
make dagster-ui

# Run full pipeline (all assets)
make run-full

# Run specific stages
make run-internal-preprocessing
make run-internal-ml
make run-external-preprocessing
make run-external-ml
make run-merging
```

#### Using Dagster Directly

```bash
# Start the Dagster UI
uv run -m clustering.dagster.app --env dev

# Run a specific Dagster job
uv run -m dagster job execute -m clustering.dagster.definitions -j internal_preprocessing_job
```

### Development Workflow

```bash
# Format, lint and type-check code
make format lint type-check

# Run tests
make test

# Clean up artifacts
make clean
```

## 🧩 Data Flow

The pipeline processes data through several stages:

1. **Data Preprocessing**
   - Internal sales data normalization
   - External data preparation

2. **Feature Engineering**
   - Feature filtering and selection
   - Missing value imputation
   - Normalization and scaling
   - Outlier detection and removal
   - Dimensionality reduction

3. **Model Training**
   - Optimal cluster count determination
   - KMeans or other clustering algorithms
   - Model persistence and versioning

4. **Cluster Assignment**
   - Store assignment to optimal clusters
   - Assignment persistence

5. **Merging**
   - Combining internal and external clusters
   - Small cluster reassignment to nearest large clusters
   - Final cluster output

## 🔧 Configuration

Create YAML configuration files in the `configs/` directory for environment-specific settings:

```yaml
# Example configuration
job_params:
  # Feature engineering parameters
  normalize: true
  norm_method: "robust"
  imputation_type: "simple"
  
  # Model training parameters
  algorithm: "kmeans"
  min_clusters: 2
  max_clusters: 10
```

## 📂 Project Structure

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
│           │   ├── preprocessing/  # Preprocessing assets
│           │   ├── clustering/     # Clustering assets
│           │   └── merging/        # Merging assets
│           ├── resources/  # Dagster resources
│           ├── sensors/    # Dagster sensors
│           └── schedules/  # Dagster schedules
├── tests/                # Test suite
├── data/                 # Data directory
│   ├── internal/         # Internal data
│   ├── external/         # External data
│   └── merging/          # Merged results
├── .env.example          # Example environment variables
├── Makefile              # Development and build commands
└── pyproject.toml        # Project metadata and dependencies
```

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`uv pip install -e ".[dev]"`)
4. Make your changes and ensure tests pass (`make test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📞 Contact

For questions or support regarding this project, please contact:

**Jackson Yang**  
Email: Jackson.Yang@cvshealth.com

## 📝 License

[MIT License](LICENSE)

Copyright (c) 2024 CVS Health
