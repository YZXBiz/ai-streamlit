"""Base for high-level project jobs."""

# %% IMPORTS
import abc
import types as TS
import typing as T
from typing import Any, Dict, Self, Type

import pydantic as pdt

from clustering.io import services

# %% TYPES
Locals = Dict[str, Any]  # Local job variables


# %% JOBS
class Job(abc.ABC, pdt.BaseModel, strict=True, extra="forbid"):
    """Base class for a job.

    This is an abstract base class for all jobs in the clustering package.
    Jobs should be created as frozen instances (with frozen=True in the class declaration)
    to maintain immutability.

    A job provides context management for running operations with proper
    setup and teardown of services like logging and alerts.

    Attributes:
        KIND: String identifier for the job type
        logger_service: Service for logging
        alerts_service: Service for sending alerts
    """

    KIND: str
    logger_service: services.LoggerService = services.LoggerService()
    alerts_service: services.AlertsService = services.AlertsService()

    def __enter__(self) -> Self:
        """Enter the job context.

        Starts the logger and alerts services.

        Returns:
            Self: The job instance
        """
        self.logger_service.start()
        logger = self.logger_service.logger()
        logger.debug("[START] Logger service: {}", self.logger_service)
        logger.debug("[START] Alerts service: {}", self.alerts_service)
        self.alerts_service.start()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TS.TracebackType | None,
    ) -> T.Literal[False]:
        """Exit the job context.

        Stops the logger and alerts services.

        Args:
            exc_type: Exception type if an exception was raised
            exc_value: Exception instance if an exception was raised
            exc_traceback: Traceback if an exception was raised

        Returns:
            False: Always propagate exceptions
        """
        logger = self.logger_service.logger()
        logger.debug("[STOP] Alerts service: {}", self.alerts_service)
        self.alerts_service.stop()
        logger.debug("[STOP] Logger service: {}", self.logger_service)
        self.logger_service.stop()
        return False  # re-raise

    @abc.abstractmethod
    def run(self) -> Locals:
        """Run the job.

        This should be implemented by subclasses to define the job's behavior.

        Returns:
            Locals: Local variables from the job execution
        """
        pass
