"""Define a job for training and registering a single AI/ML model."""

# %% IMPORTS
import typing as T
import pydantic as pdt
from clustering.io import datasets, services
from clustering.jobs import base
from clustering.core import models
from clustering.utils import helpers


# %% JOBS
class ExternalTrainingJob(base.Job):
    KIND: T.Literal["ExternalTrainingJob"] = "ExternalTrainingJob"

    # Log (assign log file name)
    logger_service: services.LoggerService = services.LoggerService(sink="logs/external_log/ext_clustering.log")

    # Data
    input_features: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    # Model
    model: models.ModelKind = pdt.Field(models.ClusteringModel(), discriminator="KIND")

    # Save
    results: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    # DVC
    dvc_lineage: dict

    def run(self) -> base.Locals:
        # services
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)

        # Read features
        logger.info("Read features: {}", self.input_features)
        inputs_ = self.input_features.read()

        # Run clustering model
        logger.info("Run clustering model")
        self.model.fit(inputs_)
        logger.info("Model trained: {}", self.model.KIND)

        # Assign clusters
        df_assigned_cluster = self.model.assign()
        logger.info("Clusters assigned back to original dataframe")
        logger.info(f"Number of clusters assigned: {df_assigned_cluster['Cluster'].nunique()}")

        # Evaluate model
        logger.info("Save clustering performance score")
        score = self.model.evaluate()
        logger.info("Clustering performance score: {}", score)

        # Save results
        logger.info("Save clustering results")
        self.results.write(df_assigned_cluster)

        # DVC lineage
        logger.info("DVC lineage")
        helpers.handle_dvc_lineage(
            folder_path=self.results.path,
            commit_message="Update external clustering data",
            push_to_remote=self.dvc_lineage.get("push_to_remote", False),
        )

        # Alerts
        self.alerts_service.notify(
            title="External Clustering Job",
            message="Clustering model trained successfully",
        )

        return locals()
