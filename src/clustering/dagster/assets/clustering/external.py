"""External clustering assets for the clustering pipeline."""

from typing import Any, Dict

import dagster as dg
import polars as pl
from pydantic import BaseModel

from clustering.core import models


class ExternalClusteringConfig(BaseModel):
    """Configuration for external clustering."""

    algorithm: str = "kmeans"
    normalize: bool = False
    norm_method: str = "clr"
    pca_active: bool = True
    pca_components: float = 0.8
    ignore_features: list[str] = ["STORE_NBR"]
    kwargs: Dict[str, Any] = {}


@dg.asset(
    io_manager_key="io_manager",
    deps=["preprocessed_external_data"],
    compute_kind="external_clustering",
    group_name="clustering",
)
def external_clustering_model(
    context: dg.AssetExecutionContext,
    preprocessed_external_data: pl.DataFrame,
    clustering_config=dg.ResourceParam(dg.InitResourceContext),
) -> models.ClusteringModel:
    """Train clustering model on external data.

    Args:
        context: Asset execution context
        preprocessed_external_data: Preprocessed external data
        clustering_config: Configuration for clustering

    Returns:
        Trained clustering model
    """
    context.log.info("Training external clustering model")

    # Create clustering model with configuration from Dagster config
    algorithm = getattr(clustering_config, "algorithm", "kmeans")
    normalize = getattr(clustering_config, "normalize", False)
    norm_method = getattr(clustering_config, "norm_method", "clr")
    pca_active = getattr(clustering_config, "pca_active", True)
    pca_components = getattr(clustering_config, "pca_components", 0.8)
    ignore_features = getattr(clustering_config, "ignore_features", ["STORE_NBR"])
    kwargs = getattr(clustering_config, "kwargs", {})

    # Create model with the specified parameters
    model = models.ClusteringModel(
        CLUS_ALGO=algorithm,
        NORMALIZE=normalize,
        NORM_METHOD=norm_method,
        PCA_ACTIVE=pca_active,
        PCA_COMPONENTS=pca_components,
        IGNORE_FEATURES=ignore_features,
        KWARGS=kwargs,
    )

    # Convert to pandas if needed (ClusteringModel expects pandas DataFrame)
    inputs = preprocessed_external_data
    if isinstance(preprocessed_external_data, pl.DataFrame):
        inputs = preprocessed_external_data.to_pandas()

    # Fit the model
    context.log.info("Fitting external clustering model")
    model.fit(inputs)

    return model


@dg.asset(
    io_manager_key="io_manager",
    deps=["external_clustering_model", "preprocessed_external_data"],
    compute_kind="external_clustering",
    group_name="clustering",
)
def external_clusters(
    context: dg.AssetExecutionContext,
    external_clustering_model: models.ClusteringModel,
    preprocessed_external_data: pl.DataFrame,
) -> pl.DataFrame:
    """Assign cluster labels to external data.

    Args:
        context: Asset execution context
        external_clustering_model: Trained external clustering model
        preprocessed_external_data: Preprocessed external data

    Returns:
        DataFrame with cluster assignments
    """
    context.log.info("Assigning clusters to external data")

    # No need to convert the data since we don't use it directly
    # The model already has the data from the fit operation

    # Assign clusters
    clustered_data = external_clustering_model.assign()

    # Convert back to polars
    result_df = pl.from_pandas(clustered_data)

    # Log number of clusters
    context.log.info(f"Number of clusters: {result_df['Cluster'].n_unique()}")

    return result_df


@dg.asset(
    io_manager_key="io_manager",
    deps=["external_clustering_model"],
    compute_kind="external_clustering",
    group_name="clustering",
)
def external_cluster_evaluation(
    context: dg.AssetExecutionContext,
    external_clustering_model: models.ClusteringModel,
) -> Dict[str, Any]:
    """Evaluate external clustering quality.

    Args:
        context: Asset execution context
        external_clustering_model: Trained external clustering model

    Returns:
        Dictionary of evaluation metrics
    """
    context.log.info("Evaluating external clusters")

    # Evaluate model
    scores = external_clustering_model.evaluate()

    # Convert to dictionary if it's a DataFrame
    if hasattr(scores, "to_dict"):
        scores_dict = scores.to_dict()
    else:
        scores_dict = scores

    return scores_dict


@dg.asset(
    io_manager_key="io_manager",
    deps=["external_clusters"],
    compute_kind="external_clustering",
    group_name="clustering",
)
def external_clustering_output(
    context: dg.AssetExecutionContext,
    external_clusters: pl.DataFrame,
    output_clusters_writer=dg.ResourceParam(dg.InitResourceContext),
) -> None:
    """Save external clustering results.

    Args:
        context: Asset execution context
        external_clusters: External clusters data
        output_clusters_writer: Writer for output clusters
    """
    context.log.info("Saving external clustering results")

    # Check if the writer expects Pandas DataFrame
    if hasattr(output_clusters_writer, "requires_pandas") and output_clusters_writer.requires_pandas:
        output_clusters_writer.write(data=external_clusters.to_pandas())
    else:
        output_clusters_writer.write(data=external_clusters)

    return None
