# Store Clustering Pipeline - Installation Guide

This guide provides step-by-step instructions for installing and running the Store Clustering Pipeline.

## Installation Options

### 1. Install Directly as a Package (Recommended)

The simplest way to install the clustering package is directly from GitHub:

```bash
# Install using pip
pip install git+https://github.com/yourusername/clustering-dagster.git

# Or install using uv
uv add git+https://github.com/yourusername/clustering-dagster.git
```

For specific components, use:

```bash
# Install with CLI functionality
pip install "git+https://github.com/yourusername/clustering-dagster.git#egg=clustering[cli]"

# Install with dashboard functionality
pip install "git+https://github.com/yourusername/clustering-dagster.git#egg=clustering[dashboard]"

# Install with all components
pip install "git+https://github.com/yourusername/clustering-dagster.git#egg=clustering[all]"
```

You can also install a specific version:

```bash
# Install specific version using tags
pip install git+https://github.com/yourusername/clustering-dagster.git@v0.1.0

# Install from a specific branch
pip install git+https://github.com/yourusername/clustering-dagster.git@main
```

### 2. Install for Development (Contributors Only)

If you're contributing to the project, install it in development mode:

```bash
# Clone the repository
git clone https://github.com/yourusername/clustering-dagster.git
cd clustering-dagster

# Install in development mode with uv
uv add -e .

# Or with specific components
uv add -e ".[dev]"     # Include development tools
uv add -e ".[cli]"     # Include CLI functionality
uv add -e ".[dashboard]" # Include dashboard functionality
uv add -e ".[all]"     # Install everything
```

## Configuration Setup

After installation, set up the required directories and configuration:

```bash
# Create required data directories
mkdir -p data/internal data/external data/merging data/raw

# Create a .env file
cat > .env << EOF
# Base directories
DATA_DIR=./data
INTERNAL_DATA_DIR=./data/internal
EXTERNAL_DATA_DIR=./data/external
MERGING_DATA_DIR=./data/merging

# Dagster configuration
DAGSTER_HOME=~/.dagster
EOF

# Create Dagster home directory
mkdir -p ~/.dagster
```

## Basic Usage

### Verify Installation

```bash
# Check if the CLI is properly installed
clustering --version
```

### Using the CLI

```bash
# Validate a configuration file
clustering validate --config your_config.yaml

# Run a pipeline job
clustering run full_pipeline --config your_config.yaml

# Check job status
clustering status

# Export results
clustering export JOB_ID --output results.csv
```

### Running the Dashboard

If you installed the dashboard component:

```bash
# Start the dashboard
clustering dashboard --port 8501
```

## Example Configuration

Create a configuration file for your pipeline:

```yaml
# config.yaml
job:
  name: full_pipeline
  params:
    cluster_count: 5
    features:
      - feature1
      - feature2
    model_params:
      random_state: 42
```

## Common Workflows

### Running a Full Pipeline

```bash
# Run with configuration file
clustering run full_pipeline --config your_config.yaml

# Or with parameters
clustering run cluster_stores --param cluster_count=5 --param use_pca=true
```

### Export and Analyze Results

```bash
# Check status of all jobs
clustering status

# Export results to CSV
clustering export JOB_ID --output results.csv

# Export results to Excel with filters
clustering export JOB_ID --output results.xlsx --format excel --filter "cluster_id > 2"
```

## Advanced Usage

### Environment-Specific Configuration

```bash
# Run in different environments
clustering run full_pipeline --env dev --config dev_config.yaml
clustering run full_pipeline --env prod --config prod_config.yaml
```

## Uninstallation

To remove the package:

```bash
# Using pip
pip uninstall clustering

# Using uv
uv remove clustering

# Remove all related configuration (optional)
rm -rf ~/.clustering
rm -rf ~/.dagster
```

## Troubleshooting

If you encounter issues:

1. Verify your configuration files are valid with `clustering validate --config your_config.yaml`
2. Check your data files with `clustering validate --data your_data.csv --schema schema_name`
3. Ensure all required directories exist
4. Check job logs with `clustering status JOB_ID`

## Configuration Guide

This section provides detailed examples of how to configure the clustering pipeline for various scenarios.

### Complete Configuration Example

Below is a comprehensive configuration file showing all available options:

```yaml
# config.yaml - Complete configuration example
paths:
  base_data_dir: ${env:DATA_DIR,./data}
  internal_data_dir: ${env:INTERNAL_DATA_DIR,./data/internal}
  external_data_dir: ${env:EXTERNAL_DATA_DIR,./data/external}
  merging_data_dir: ${env:MERGING_DATA_DIR,./data/merging}
  output_dir: ${env:OUTPUT_DIR,./output}

data:
  internal:
    sales_file: "internal/sales_data.csv"
    product_file: "internal/product_hierarchy.csv"
    store_attributes_file: "internal/store_attributes.csv"
    date_range:
      start_date: "2023-01-01"
      end_date: "2023-12-31"
  external:
    demographics_file: "external/demographics.csv"
    competitors_file: "external/competitors.csv"
    geo_data_file: "external/geo_data.csv"

preprocessing:
  normalize: true
  normalization_method: "robust"  # Options: standard, minmax, robust
  impute_missing: true
  imputation_strategy: "median"   # Options: mean, median, constant, knn
  outlier_removal: true
  outlier_threshold: 3.0          # Z-score threshold for outlier detection
  feature_selection: true
  feature_importance_threshold: 0.05
  pca:
    enabled: true
    n_components: 0.95           # Can be int or float (variance explained)
    random_state: 42

model:
  algorithm: "kmeans"            # Options: kmeans, hierarchical, dbscan
  params:
    kmeans:
      init: "k-means++"
      n_init: "auto"
      random_state: 42
    hierarchical:
      linkage: "ward"
      affinity: "euclidean"
    dbscan:
      eps: 0.5
      min_samples: 5
  cluster_selection:
    method: "auto"              # Options: auto, manual
    min_clusters: 3
    max_clusters: 12
    optimal_metric: "silhouette" # Options: silhouette, calinski_harabasz, davies_bouldin
  
merging:
  method: "weighted"            # Options: weighted, dominant, ensemble
  weights:
    internal: 0.7
    external: 0.3

output:
  format: "csv"                 # Options: csv, excel, parquet
  include_intermediates: false
  visualizations: true
  reports: true

compute:
  memory_optimized: false
  parallel_processing: true
  n_jobs: -1                    # -1 uses all available cores
```

### Common Configuration Scenarios

#### 1. Basic Clustering with Minimal Configuration

```yaml
# minimal_config.yaml
paths:
  base_data_dir: ./data

data:
  internal:
    sales_file: "internal/sales_data.csv"
    store_attributes_file: "internal/store_attributes.csv"

preprocessing:
  normalize: true

model:
  algorithm: "kmeans"
  params:
    kmeans:
      random_state: 42
  cluster_selection:
    min_clusters: 3
    max_clusters: 8
```

To run with this configuration:

```bash
clustering run internal_ml_job --config minimal_config.yaml
```

#### 2. Production Configuration with Advanced Options

```yaml
# production_config.yaml
paths:
  base_data_dir: /opt/data/clustering
  output_dir: /opt/data/clustering/output

data:
  internal:
    sales_file: "internal/sales_full_year.csv"
    product_file: "internal/product_hierarchy_v2.csv"
    date_range:
      start_date: "2023-01-01"
      end_date: "2023-12-31"
  external:
    demographics_file: "external/census_demographics.csv"
    competitors_file: "external/competitor_locations.csv"

preprocessing:
  normalize: true
  normalization_method: "robust"
  impute_missing: true
  outlier_removal: true
  pca:
    enabled: true
    n_components: 0.9

model:
  algorithm: "kmeans"
  params:
    kmeans:
      n_init: 10
      random_state: 42
  cluster_selection:
    method: "auto"
    min_clusters: 4
    max_clusters: 15

compute:
  memory_optimized: true
  parallel_processing: true
  n_jobs: 8
```

To run with this configuration:

```bash
clustering run full_pipeline_job --config production_config.yaml --env prod
```

## Python SDK Usage

For users who want to integrate the clustering pipeline into their own Python applications, this section provides examples of using the package programmatically.

### Basic SDK Usage

```python
from clustering.pipeline.definitions import define_asset_job
from dagster import execute_job

# Execute a complete job with configuration
result = execute_job(
    define_asset_job("full_pipeline_job"),
    run_config={
        "ops": {
            "preprocess_internal_data": {
                "config": {
                    "normalize": True,
                    "impute_missing": True,
                }
            },
            "train_clustering_model": {
                "config": {
                    "algorithm": "kmeans",
                    "n_clusters": 5,
                    "random_state": 42
                }
            }
        }
    }
)

# Access results
for event in result.events_for_node("train_clustering_model"):
    if event.event_type_value == "ASSET_MATERIALIZATION":
        model_path = event.event_specific_data.materialization.metadata_entries[0].entry_data.path
        print(f"Model saved to: {model_path}")
```

### Working with Individual Components

```python
import pandas as pd
from clustering.shared.preprocessing import normalize_features, remove_outliers
from clustering.shared.models import ClusteringModel

# Load and preprocess your data
data = pd.read_csv("my_data.csv")

# Preprocessing
normalized_data = normalize_features(
    data, 
    method="robust", 
    columns=["feature1", "feature2", "feature3"]
)
clean_data = remove_outliers(normalized_data, threshold=3.0)

# Model training
model = ClusteringModel(algorithm="kmeans", n_clusters=5, random_state=42)
model.fit(clean_data)

# Get cluster assignments
clusters = model.predict(clean_data)
data["cluster"] = clusters

# Analyze clusters
from clustering.shared.analysis import calculate_cluster_metrics, generate_cluster_profiles

metrics = calculate_cluster_metrics(data, clusters)
profiles = generate_cluster_profiles(data, clusters)

print(f"Silhouette Score: {metrics['silhouette_score']}")
print(f"Cluster Profiles:\n{profiles}")
```

### Advanced Configuration

```python
from clustering.pipeline.resources import configure_resources
from clustering.pipeline.assets import run_full_pipeline
from clustering.shared.config import ClusteringConfig
from pathlib import Path

# Load configuration from file
config_path = Path("config.yaml")
config = ClusteringConfig.from_yaml(config_path)

# Or create programmatically
config = ClusteringConfig(
    paths={
        "base_data_dir": Path("./data"),
        "output_dir": Path("./output")
    },
    preprocessing={
        "normalize": True,
        "impute_missing": True,
        "pca": {"enabled": True, "n_components": 0.95}
    },
    model={
        "algorithm": "kmeans",
        "params": {"kmeans": {"random_state": 42}},
        "cluster_selection": {"min_clusters": 3, "max_clusters": 10}
    }
)

# Set up resources
resources = configure_resources(config)

# Run pipeline with custom configuration
results = run_full_pipeline(
    sales_data_path="my_custom_data.csv",
    config=config,
    resources=resources
)

# Access and use results
optimal_clusters = results["optimal_cluster_count"]
cluster_assignments = results["cluster_assignments"]
cluster_profiles = results["cluster_profiles"]

print(f"Found {optimal_clusters} optimal clusters")
print(f"Cluster distribution: {cluster_assignments['cluster'].value_counts()}")
```

### Integration with Other Frameworks

```python
import pandas as pd
from clustering.shared.models import ClusteringModel
from clustering.shared.preprocessing import preprocess_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Use clustering as a feature engineering step in an ML pipeline
data = pd.read_csv("customer_data.csv")
features = preprocess_features(data, normalize=True, impute=True)

# Generate cluster labels
cluster_model = ClusteringModel(algorithm="kmeans", n_clusters=5)
data["cluster"] = cluster_model.fit_predict(features)

# Use clusters as a feature for a downstream model
X = pd.concat([features, pd.get_dummies(data["cluster"], prefix="cluster")], axis=1)
y = data["target_variable"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

accuracy = classifier.score(X_test, y_test)
print(f"Model accuracy with clustering features: {accuracy:.2f}")
``` 