"""Clustering pipeline for internal data."""

import typing as T
from typing import Any, cast

import polars as pl
import pydantic as pdt

from clustering.core import models
from clustering.io import datasets, services
from clustering.jobs import base
from clustering.jobs.utils import get_path_safely, track_dvc_lineage


# %% JOBS
class InternalTrainingJob(base.Job):
    """Internal training job.

    This job handles the clustering of internal sales data by product category.
    It loads preprocessed sales data, trains models for each category, and
    produces cluster assignments for each store/category pair.
    """

    KIND: T.Literal["InternalTrainingJob"] = "InternalTrainingJob"

    default_artifact_path: str = "artifacts/internal_training"

    logger_service: services.LoggerService = services.LoggerService(sink="logs/internal_log/int_train.log")

    # Data
    inputs: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    # ML
    model: models.ModelKind = pdt.Field(..., discriminator="KIND")

    # Save
    outputs: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    # DVC
    dvc_lineage: dict[str, Any] = pdt.Field(default_factory=dict)

    def run(self) -> base.Locals:
        """Run the clustering pipeline in logical steps."""
        logger = self.logger_service.logger()
        logger.info("Starting clustering pipeline")

        # Step 1: Load input data
        logger.info("Loading input data")
        input_data = cast(dict[str, pl.DataFrame], self.inputs.read())

        # Step 2: Train models for each category
        logger.info("Training models")
        results = {}
        eval_scores = {}

        for category, df in input_data.items():
            logger.info(f"Processing category: {category}")
            if not isinstance(df, pl.DataFrame):
                df = pl.from_pandas(df)

            # Train model and generate assignments
            self.model.fit(df)
            df_assigned = cast(pl.DataFrame, self.model.assign(df))

            # Check if cluster column exists
            try:
                if isinstance(df_assigned, pl.DataFrame) and "Cluster" in df_assigned.columns:
                    n_clusters = df_assigned.select(pl.col("Cluster").n_unique()).item()
                    logger.info(f"Assigned {n_clusters} unique clusters")
                else:
                    logger.warning("Cluster column not found in assigned DataFrame")
            except Exception as e:
                logger.warning(f"Could not determine number of clusters: {e}")

            # Store results
            results[category] = df_assigned

            # Evaluate model
            try:
                score = cast(float, self.model.evaluate())
                eval_scores[category] = score
                logger.info(f"Model evaluation score: {score:.4f}")
            except Exception as e:
                logger.warning(f"Could not evaluate model: {e}")
                eval_scores[category] = float("nan")

        # Step 3: Save results
        logger.info("Saving results")

        # Check if the writer expects Pandas DataFrame
        if hasattr(self.outputs, "requires_pandas") and self.outputs.requires_pandas:
            self.outputs.write(data={k: df.to_pandas() for k, df in results.items()})
        else:
            self.outputs.write(data=results)

        # Step 4: Track lineage
        logger.info("Tracking lineage")
        self._track_lineage()

        # Step 5: Send notification
        logger.info("Sending notification")
        self._send_notification(eval_scores)

        return locals()

    def _track_lineage(self) -> None:
        """Track data lineage with DVC."""
        logger = self.logger_service.logger()
        push_remote = self.dvc_lineage.get("push_to_remote", False)

        logger.info("Tracking DVC lineage")
        # Track inputs
        track_dvc_lineage(self.inputs, "Update internal inputs", push_remote, logger)

        # Track outputs
        track_dvc_lineage(self.outputs, "Update internal outputs", push_remote, logger)

    def _send_notification(self, eval_scores: dict[str, float]) -> None:
        """Send notification about job completion."""
        logger = self.logger_service.logger()
        output_path = get_path_safely(self.outputs)

        # Create summary message
        score_msg = "\n".join([f"{cat}: {score:.4f}" for cat, score in eval_scores.items()])
        message = f"Clustering complete. Results saved to {output_path}.\n\nEvaluation scores:\n{score_msg}"

        logger.info("Sending notification")
        self.alerts_service.notify(
            title="Internal Clustering Done",
            message=message,
        )
