"""Data cleaning. ETL pipeline."""

# %% IMPORTS
import typing as T
from typing_extensions import Annotated

import pydantic as pdt

from clustering.io import services, datasets
from clustering.jobs import base

from clustering.utils import helpers
from clustering.utils.helpers import merge_dataframes

# %% TYPES
# Annotate so it can be used in a list
InputItem = Annotated[datasets.ReaderKind, pdt.Field(discriminator="KIND")]


# %% JOBS
class ExternalPreproJob(base.Job):
    KIND: T.Literal["ExternalPreproJob"] = "ExternalPreproJob"

    # Log (assign log file name)
    logger_service: services.LoggerService = services.LoggerService(sink="logs/external_log/ext_preprocessing.log")

    # Data
    # - External
    input_external: list[InputItem]

    # Save
    output_data: datasets.WriterKind = pdt.Field(..., discriminator="KIND")

    # DVC
    dvc_lineage: dict

    def run(self) -> base.Locals:
        # services
        # - logger
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)

        # Read and merge all external dataframes
        df_list = []
        for input_data in self.input_external:
            logger.info("Read external features: {}", input_data)
            df_list.append(input_data.read())

        logger.info("Merge all above read dataframes on STORE_NBR")
        df_merged = merge_dataframes(df_list)

        logger.info("Save final features")
        self.output_data.write(df_merged)

        # DVC lineage
        logger.info("DVC lineage")
        helpers.handle_dvc_lineage(
            folder_path=self.output_data.path,
            commit_message="Update external prepro data",
            push_to_remote=self.dvc_lineage.get("push_to_remote", False),
        )

        self.alerts_service.notify(
            title="Extenral Preprocessing Done",
            message=f"Clustering done. Saved to {self.output_data.path}.",
        )

        return locals()
