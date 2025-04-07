"""Define trainable clustering models."""
# Author: Jackson Yang (Jackson.Yang@cvshealth.com)

# %% IMPORTS

import logging
import typing as tp
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pydantic as pdt
from pycaret.clustering import ClusteringExperiment
from pydantic import PrivateAttr
from sklearn.cluster import KMeans
from typing_extensions import Self  # Use typing_extensions for Python 3.10 compatibility
from yellowbrick.cluster import KElbowVisualizer

from clustering.core import schemas

# %% Set up module-level logger
logger = logging.getLogger(__name__)


# %% MODELS
class Model(ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model.

    Use a model to adapt AI/ML frameworks.
    e.g., to swap easily one model with another.
    """

    KIND: str

    def get_params(self) -> dict[str, tp.Any]:
        """Get the model params.

        Returns:
        -------
            Dict containing model parameters

        """
        return {
            key: value
            for key, value in self.model_dump().items()
            if not key.startswith("_") and not key.isupper()
        }

    def set_params(self, **params: object) -> "Model":
        """Set the model params in place.

        Args:
        ----
            **params: Parameter key-value pairs to set

        Returns:
        -------
            Self instance with updated parameters

        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abstractmethod
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets | None = None) -> Self:
        """Fit the model on the given inputs and targets.

        Args:
        ----
            inputs: model training inputs
            targets: model training targets (optional for unsupervised learning)

        Returns:
        -------
            instance of the model

        """

    @abstractmethod
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs | int:
        """Generate outputs with the model for the given inputs.

        Args:
        ----
            inputs: model prediction inputs

        Returns:
        -------
            model prediction outputs or cluster assignment

        """


# %% TRANSFORMER REGISTRY
class ClusteringModel(Model):
    """Underlying model for clustering leveraging PyCaret.

    This model uses PyCaret's ClusteringExperiment to handle the entire
    clustering pipeline including preprocessing, dimensionality reduction,
    model training, and evaluation.

    Attributes:
    ----------
        KIND: Type of the model. Default is "ClusteringModel".
        IGNORE_FEATURES: Features to ignore. Default is ["STORE_NBR"].
        CLUS_ALGO: Clustering algorithm to use. Default is "kmeans".
            - kmeans
            - ap (Affinity Propagation)
            - meanshift
            - sc (Spectral Clustering)
            - hclust (Hierarchical Clustering)
            - dbscan
            - optics
            - birch
        NORMALIZE: Whether to normalize the data. Default is False.
        NORM_METHOD: Normalization method to use. Default is "clr".
            - zscore
            - minmax
            - maxabs
            - robust
            - clr (Centered Log Ratio)
        PCA_ACTIVE: Whether to apply PCA. Default is True.
        PCA_COMPONENTS: Number of PCA components to use. Default is 0.8.
            - int: number of components
            - float: percentage of variance to explain
            - None: no PCA
        PCA_METHOD: PCA method to use. Default is "linear".
            - linear
            - kernel
            - incremental
        KWARGS: Additional model parameters. Default is an empty dict.

    """

    KIND: tp.Literal["ClusteringModel"] = "ClusteringModel"

    # Feature engineering
    IGNORE_FEATURES: list[str] | None = ["STORE_NBR"]
    NORMALIZE: bool = False
    NORM_METHOD: tp.Literal["zscore", "minmax", "maxabs", "robust", "clr"] = "clr"
    PCA_ACTIVE: bool = True
    PCA_COMPONENTS: int | float = 0.8
    PCA_METHOD: tp.Literal["linear", "kernel", "incremental"] = "linear"

    # Model parameters
    CLUS_ALGO: tp.Literal[
        "kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"
    ] = "kmeans"

    # Additional model parameters will be added via KWARGS
    KWARGS: dict[str, tp.Any] = {}

    # Private attributes
    _clus_exp: ClusteringExperiment = PrivateAttr(default=None)
    _model: tp.Any = PrivateAttr(default=None)
    _optimal_clusters: int | None = PrivateAttr(default=None)

    def _raise_if_model_not_fit(self, message: str) -> None:
        """Raise an error if the model has not been fit.

        Args:
        ----
            message: Error message to use in the ValueError

        Raises:
        ------
            ValueError: Always raised if model is not fit

        """
        if self._model is None or self._clus_exp is None:
            raise ValueError(message)

    def fit(
        self,
        inputs: schemas.Inputs,
        # Unused parameter kept for API compatibility
        _targets: schemas.Targets | None = None,
    ) -> "ClusteringModel":
        """Fit the clustering model to input data.

        Args:
        ----
            inputs: DataFrame containing features to cluster
            _targets: Not used for clustering (included for API compatibility)

        Returns:
        -------
            Self with fitted model

        Raises:
        ------
            ValueError: If optimal number of clusters cannot be determined
            Exception: For any other errors during model training

        """
        try:
            logger.info("Initializing clustering experiment with algorithm: %s", self.CLUS_ALGO)
            self._clus_exp = ClusteringExperiment()

            # Set up the experiment
            self._clus_exp.setup(
                data=inputs,
                verbose=True,
                ignore_features=self.IGNORE_FEATURES,
                pca=self.PCA_ACTIVE,
                pca_method=self.PCA_METHOD,  # ignored when pca is False
                pca_components=self.PCA_COMPONENTS,  # ignored when pca is False
                normalize=self.NORMALIZE,
                normalize_method=self.NORM_METHOD,
                # mlflow settings
                system_log=False,
                log_experiment=False,  # * disable mlflow logging for clustering
                experiment_name="ClusteringExperiment",
                # additional kwargs
                **self.KWARGS,
            )

            logger.info("Finding optimal number of clusters...")
            # Determine the optimal number of clusters using Yellowbrick
            self._optimal_clusters = self._find_optimal_clusters(self._clus_exp.X_train_transformed)

            if self._optimal_clusters is None or self._optimal_clusters <= 1:
                error_msg = (
                    f"The number of clusters must be greater than 1; "
                    f"The optimal number of clusters is {self._optimal_clusters}"
                )
                self._raise_cluster_value_error(error_msg)

            logger.info("Creating model with %s clusters", self._optimal_clusters)
            # Create the model with the optimal number of clusters
            self._model = self._clus_exp.create_model(
                estimator=self.CLUS_ALGO,
                num_clusters=self._optimal_clusters,
                random_state=self.KWARGS.get("random_state", 42),  # Add reproducibility
            )
        except Exception:
            logger.exception("Error in clustering model fitting")
            raise
        else:
            return self

    def _raise_cluster_value_error(self, message: str) -> tp.NoReturn:
        """Raise a ValueError with the given message.

        Args:
        ----
            message: Error message

        Raises:
        ------
            ValueError: Always raised

        """
        raise ValueError(message)

    def _find_optimal_clusters(self, data: np.ndarray) -> int | None:
        """Use Yellowbrick to find the optimal number of clusters.

        Args:
        ----
            data: Transformed data to use for finding optimal clusters

        Returns:
        -------
            The optimal number of clusters or None if it cannot be determined

        """
        try:
            # Use fixed random state for reproducibility
            model = KMeans(random_state=self.KWARGS.get("random_state", 42))

            # Limit the range to reasonable values based on data size
            max_k = min(15, len(data) // 5)
            min_k = 2

            if max_k <= min_k:
                logger.warning("Not enough samples to determine optimal clusters")
                return 2  # Default to minimum clusters

            visualizer = KElbowVisualizer(model, k=(min_k, max_k))
            visualizer.fit(data)

            if visualizer.elbow_value_ is None:
                logger.warning("Could not determine elbow point, using default of 3 clusters")
                return 3
        except Exception:
            logger.exception("Error finding optimal clusters")
            # Return a default value rather than failing
            return 3
        else:
            return visualizer.elbow_value_

    def assign(self) -> schemas.Outputs:
        """Assign cluster labels to the input data, given a trained model.

        Returns:
        -------
            DataFrame with cluster assignments

        Raises:
        ------
            ValueError: If model has not been fit

        """
        self._raise_if_model_not_fit("Model must be fit before assigning clusters")

        result = self._clus_exp.assign_model(self._model)

        # Add back ignored features only if they exist
        if self.IGNORE_FEATURES:
            for feature in self.IGNORE_FEATURES:
                if feature in self._clus_exp.data.columns:
                    result[feature] = self._clus_exp.data[
                        feature
                    ]  # the index is the same as the original data

        return result

    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        """Predict cluster labels for new data.

        Args:
        ----
            inputs: New data to predict cluster assignments for

        Returns:
        -------
            DataFrame with cluster predictions

        Raises:
        ------
            ValueError: If model has not been fit

        """
        self._raise_if_model_not_fit("Model must be fit before making predictions")
        return self._clus_exp.predict_model(self._model, data=inputs)

    def evaluate(self) -> schemas.Outputs:
        """Evaluate the clustering performance.

        Returns:
        -------
            DataFrame with evaluation metrics

        Raises:
        ------
            ValueError: If model has not been fit

        """
        self._raise_if_model_not_fit("Model must be fit before evaluation")
        return self._clus_exp.pull()

    def save(self, path: str | Path | None = None) -> None:
        """Save the model to the specified path or current directory.

        Args:
        ----
            path: Directory path to save the model (default: current directory)

        Raises:
        ------
            ValueError: If model has not been fit

        """
        self._raise_if_model_not_fit("Model must be fit before saving")

        try:
            save_path = path or "."
            model_name = f"{self.KIND}_{self._optimal_clusters}_clusters"

            logger.info("Saving model to %s as %s", save_path, model_name)
            self._clus_exp.save_model(self._model, model_name=model_name)
        except Exception:
            logger.exception("Error saving model")
            raise


# Type alias for backwards compatibility
ModelKind = ClusteringModel
