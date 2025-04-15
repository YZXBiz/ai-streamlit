"""Logging service for the clustering pipeline."""

from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import loguru
import pydantic as pdt


class LoggerService(pdt.BaseModel):
    """Service for logging messages.

    https://loguru.readthedocs.io/en/stable/api/logger.html

    Attributes:
        sink: Logging output destination.
        level: Logging level.
        format: Logging format.
        colorize: Whether to colorize output.
        serialize: Whether to convert to JSON.
        backtrace: Whether to enable exception trace.
        diagnose: Whether to enable variable display.
        catch: Whether to catch errors during log handling.
    """

    sink: str = "logs/app.log"  # Save logs to a file in the logs folder
    level: str = "DEBUG"
    format: str = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {message}"
    colorize: bool = True
    serialize: bool = False
    backtrace: bool = True
    diagnose: bool = False
    catch: bool = True

    class model_config:
        """Pydantic model configuration."""
        extra = "forbid"

    def start(self) -> None:
        """Start the logging service."""
        loguru.logger.remove()
        config = self.model_dump()
        # Ensure the logs directory exists
        log_dir = Path(config["sink"]).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        # Attach the latest timestamp to the sink name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{Path(config['sink']).stem}_{timestamp}{Path(config['sink']).suffix}"
        config["sink"] = str(log_file)
        loguru.logger.add(**config)

    def stop(self) -> None:
        """Stop the logging service."""
        # Nothing to do as loguru handles this automatically
        pass

    def logger(self) -> "loguru.Logger":
        """Return the main logger.

        Returns:
            The configured logger instance.
        """
        return loguru.logger
