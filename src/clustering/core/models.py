"""Define trainable clustering models."""
# Author: Jackson Yang (Jackson.Yang@cvshealth.com)

# %% IMPORTS

import typing as T
from yellowbrick.cluster import KElbowVisualizer
from clustering.core import schemas
from pycaret.clustering import ClusteringExperiment
from pydantic import PrivateAttr
from sklearn.cluster import KMeans
import pydantic as pdt
from abc import ABC, abstractmethod


# %% MODELS
class Model(ABC, pdt.BaseModel, strict=True, frozen=False, extra="forbid"):
    """Base class for a project model.

    Use a model to adapt AI/ML frameworks.
    e.g., to swap easily one model with another.
    """

    KIND: str

    def get_params(self, deep: bool = True) -> dict[str, T.Any]:
        """Get the model params.

        Args:
            deep (bool, optional): ignored.

        Returns:
            Params: internal model parameters.
        """
        params: dict[str, T.Any] = {}
        for key, value in self.model_dump().items():
            if not key.startswith("_") and not key.isupper():
                params[key] = value
        return params

    def set_params(self, **params: T.Any) -> T.Self:
        """Set the model params in place.

        Returns:
            T.Self: instance of the model.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    @abstractmethod
    def fit(self, inputs: schemas.Inputs, targets: schemas.Targets | None) -> T.Self:
        """Fit the model on the given inputs and targets.

        Args:
            inputs (schemas.Inputs): model training inputs.
            targets (schemas.Targets): model training targets.
                - targets are optional for unsupervised learning.

        Returns:
            T.Self: instance of the model.
        """

    @abstractmethod
    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs | int:
        """Generate outputs with the model for the given inputs.

        Args:
            inputs (schemas.Inputs): model prediction inputs.

        Returns:
            schemas.Outputs: model prediction outputs.
            int: unsupervised learning cluster.
        """


# %% TRANSFORMER REGISTRY
class ClusteringModel(Model):
    """
    Underlying model for clustering leveraging PyCaret.

    Attributes:
        KIND (str): Type of the model. Default is "BaseClusteringModel".
        IGNORE_FEATURES (str): Features to ignore. Default is "STORE_NBR".
        CLUS_ALGO (str): Clustering algorithm to use. Default is "kmeans".
            - kmeans
            - ap (Affinity Propagation)
            - meanshift
            - sc (Spectral Clustering)
            - hclust (Hierarchical Clustering)
            - dbscan
            - optics
            - birch
        NORMALIZE (bool): Whether to normalize the data. Default is True.
        NORM_METHOD (str): Normalization method to use. Default is "zscore".
            - zscore
            - minmax
            - maxabs
            - robust
        CUSTOM_SCALER (str): Custom scaler to use. Default is "clr".
            - clr (Centered Log Ratio)
        CUSTOM_PIPELINE_POSITION (int): Position of the custom pipeline. Default is 0.
        PCA_ACTIVE (bool): Whether to apply PCA. Default is True.
        PCA_METHOD (str): PCA method to use. Default is "linear".
            - linear
            - kernel
            - incremental
        PCA_COMPONENTS (int): Number of PCA components to use. Default is 0.8.
            - int: number of components
            - float: percentage of variance to explain
            - None: no PCA
        KWARGS (dict): Additional model parameters. Default is an empty dict.
    """

    KIND: T.Literal["ClusteringModel"] = "ClusteringModel"

    # Feature engineering
    IGNORE_FEATURES: list[str] | None = ["STORE_NBR"]
    NORMALIZE: bool = False
    NORM_METHOD: T.Literal["zscore", "minmax", "maxabs", "robust", "clr"] = "clr"
    PCA_ACTIVE: bool = True
    PCA_COMPONENTS: int | float = 0.8
    PCA_METHOD: T.Literal["linear", "kernel", "incremental"] = "linear"

    # Model parameters
    CLUS_ALGO: T.Literal["kmeans", "ap", "meanshift", "sc", "hclust", "dbscan", "optics", "birch"] = "kmeans"

    # Additional model parameters will be added via KWARGS
    KWARGS: dict[str, T.Any] = {}

    # Model Save Path
    _clus_exp: ClusteringExperiment = PrivateAttr()
    _model: T.Any = PrivateAttr()

    def fit(
        self,
        inputs: schemas.Inputs,
    ) -> "ClusteringModel":
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

        # Determine the optimal number of clusters using Yellowbrick
        optimal_clusters = self._find_optimal_clusters(self._clus_exp.X_train_transformed)  # used transformed data

        # TODO: clustering now is still random, need to set random_state
        if optimal_clusters is None or optimal_clusters <= 1:
            raise ValueError(
                f"The number of clusters must be greater than 1; The optimal number of clusters is {optimal_clusters}"
            )

        # Create the model with the optimal number of clusters
        self._model = self._clus_exp.create_model(estimator=self.CLUS_ALGO, num_clusters=optimal_clusters)

        return self

    def _find_optimal_clusters(self, data):
        """Use Yellowbrick to find the optimal number of clusters."""
        model = KMeans()
        visualizer = KElbowVisualizer(model, k=(2, 15))
        visualizer.fit(data)
        return visualizer.elbow_value_

    def assign(self) -> schemas.Outputs:
        """Assign cluster labels to the input data, given a trained model."""
        result = self._clus_exp.assign_model(self._model)

        # Add back ignored features only if they exist
        for feature in self.IGNORE_FEATURES:
            if feature in self._clus_exp.data.columns:
                result[feature] = self._clus_exp.data[feature]  # the index is the same as the original data

        return result

    def predict(self, inputs: schemas.Inputs) -> schemas.Outputs:
        """This method is used to predict the cluster labels for new data."""
        predictions = self._clus_exp.predict_model(self._model, data=inputs)
        return predictions

    def evaluate(self) -> schemas.Outputs:
        """Evaluate the clustering performance."""
        score = self._clus_exp.pull()
        return score

    def save(self) -> None:
        """Save the model to the current path."""
        # TODO: need to find a way to add this directly to remote Azure Blob Storage
        self._clus_exp.save_model(self._model, model_name=self.KIND)  # save model to current path


ModelKind = ClusteringModel
