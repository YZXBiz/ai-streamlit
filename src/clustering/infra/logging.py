"""Logging service for the clustering pipeline."""

from datetime import datetime
from pathlib import Path

import loguru
import pydantic as pdt


class LoggerService(pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Service for logging messages.

    https://loguru.readthedocs.io/en/stable/api/logger.html

    Parameters:
        sink (str): logging output.
        level (str): logging level.
        format (str): logging format.
        colorize (bool): colorize output.
        serialize (bool): convert to JSON.
        backtrace (bool): enable exception trace.
        diagnose (bool): enable variable display.
        catch (bool): catch errors during log handling.
    """

    sink: str = "logs/app.log"  # Save logs to a file in the logs folder
    level: str = "DEBUG"
    format: str = "[{time:YYYY-MM-DD HH:mm:ss}] [{level}] {message}"
    colorize: bool = True
    serialize: bool = False
    backtrace: bool = True
    diagnose: bool = False
    catch: bool = True

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

    def logger(self) -> loguru.Logger:
        """Return the main logger.

        Returns:
            loguru.Logger: the main logger.
        """
        return loguru.logger
