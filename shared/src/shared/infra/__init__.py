"""Infrastructure utilities for the clustering project."""

from enum import Enum
from shared.infra.hydra_config import load_config
from shared.infra.logging import LoggerService


class Environment(str, Enum):
    """Environment enum for configuration."""
    
    DEV = "dev"
    TEST = "test"
    STAGING = "staging"
    PROD = "prod"
    
    def __str__(self) -> str:
        """Convert to string for configuration usage.
        
        Returns:
            Environment value as string
        """
        return self.value


__all__ = ["Environment", "load_config", "LoggerService"]
