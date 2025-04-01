"""Data cleaning and ETL pipeline for internal data."""

# %% IMPORTS
import typing as T
from typing import Any, Dict

import polars as pl
import pydantic as pdt

from clustering.core import schemas
from clustering.io import datasets, services
from clustering.jobs import base
from clustering.jobs.utils import get_path_safely, track_dvc_lineage, validate_dataframe
from clustering.utils.data_processing import clean_ns, create_cat_dict, distribute_sales_evenly, merge_sales_ns


# %% JOBS
class InternalPreproJob(base.Job):
    """Internal preprocessing job.

    This job handles preprocessing of internal sales and need state data.
    It cleans the data, merges sales with need states, and calculates
    sales percentages by category and need state.
    """

    KIND: T.Literal["InternalPreproJob"] = "InternalPreproJob"

    default_artifact_path: str = "artifacts/internal_preprocessing"

    logger_service: services.LoggerService = services.LoggerService(sink="logs/internal_log/int_prepro.log")

    # Data
    input_sales: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")
    input_need_state: datasets.ReaderKind = pdt.Field(..., discriminator="KIND")

    # Save
    output_sales: datasets.WriterKind = pdt.Field(..., discriminator="KIND")
    output_sales_percent: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    # DVC
    dvc_lineage: Dict[str, Any] = pdt.Field(default_factory=dict)

    def run(self) -> base.Locals:
        """Run the preprocessing pipeline in logical steps."""
        logger = self.logger_service.logger()
        logger.info("Starting preprocessing pipeline")

        # Step 1: Load and validate inputs
        inputs_data = self._load_and_validate_inputs()

        # Step 2: Process data
        processed_data = self._process_data(inputs_data)

        # Step 3: Save results
        self._save_results(processed_data)

        # Step 4: Track data lineage
        self._track_lineage()

        # Step 5: Send notification
        self._send_notification()

        return locals()

    def _load_and_validate_inputs(self) -> dict[str, pl.DataFrame]:
        """Load and validate all input data."""
        logger = self.logger_service.logger()

        # Load sales data
        logger.info("Reading sales data")
        # Convert Pandas to Polars if needed
        sales_data_raw = self.input_sales.read()
        if not isinstance(sales_data_raw, pl.DataFrame):
            sales_data = pl.from_pandas(sales_data_raw)
        else:
            sales_data = sales_data_raw

        sales_data_validated = validate_dataframe(sales_data, schemas.InputsSalesSchema, logger)

        # Load need state data
        logger.info("Reading need state data")
        ns_data_raw = self.input_need_state.read()
        if not isinstance(ns_data_raw, pl.DataFrame):
            ns_data = pl.from_pandas(ns_data_raw)
        else:
            ns_data = ns_data_raw

        ns_data_validated = validate_dataframe(ns_data, schemas.InputsNSSchema, logger)

        # Clean need state data
        logger.info("Cleaning need state data")
        ns_data_cleaned = clean_ns(ns_data_validated)

        return {"sales": sales_data_validated, "need_state": ns_data_cleaned}

    def _process_data(self, inputs: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
        """Process the input data."""
        logger = self.logger_service.logger()

        # Merge data
        logger.info("Merging sales and need state data")
        merged_data = merge_sales_ns(df_sales=inputs["sales"], df_ns=inputs["need_state"])
        merged_validated = validate_dataframe(merged_data, schemas.InputsMergedSchema, logger)

        # Redistribute sales
        logger.info("Redistributing sales based on need states")
        distributed_sales = distribute_sales_evenly(merged_validated)
        distributed_validated = validate_dataframe(distributed_sales, schemas.InputsMergedSchema, logger)

        # Create category dictionary
        logger.info("Creating category dictionary")
        category_dict = create_cat_dict(distributed_validated)

        return {"distributed_sales": distributed_validated, "category_dict": category_dict}

    def _save_results(self, data: dict[str, pl.DataFrame]) -> None:
        """Save processed data to outputs."""
        logger = self.logger_service.logger()

        # Check if the writer expects Pandas DataFrame
        logger.info("Saving preprocessed sales data")
        if hasattr(self.output_sales, "requires_pandas") and self.output_sales.requires_pandas:
            self.output_sales.write(data=data["distributed_sales"].to_pandas())
        else:
            self.output_sales.write(data=data["distributed_sales"])

        logger.info("Saving preprocessed sales percent data")
        if hasattr(self.output_sales_percent, "requires_pandas") and self.output_sales_percent.requires_pandas:
            self.output_sales_percent.write(data={k: df.to_pandas() for k, df in data["category_dict"].items()})
        else:
            self.output_sales_percent.write(data=data["category_dict"])

    def _track_lineage(self) -> None:
        """Track data lineage with DVC."""
        logger = self.logger_service.logger()
        push_remote = self.dvc_lineage.get("push_to_remote", False)

        logger.info("Tracking DVC lineage")
        # Track input need state
        track_dvc_lineage(self.input_need_state, "Update internal need state data", push_remote, logger)

        # Track output sales
        track_dvc_lineage(self.output_sales, "Update internal sales data", push_remote, logger)

        # Track output sales percent
        track_dvc_lineage(self.output_sales_percent, "Update internal sales percent data", push_remote, logger)

    def _send_notification(self) -> None:
        """Send notification about job completion."""
        logger = self.logger_service.logger()
        output_path = get_path_safely(self.output_sales_percent)

        logger.info("Sending notification")
        self.alerts_service.notify(
            title="Internal Preprocessing Done",
            message=f"Preprocessing done. Saved to {output_path} and versioned with DVC.",
        )
