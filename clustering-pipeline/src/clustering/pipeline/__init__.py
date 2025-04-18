"""Pipeline module for clustering project."""

__version__ = "1.0.0"

# Import the assets module
from clustering.pipeline import assets

# Just expose the assets sub-modules without circular imports
__all__ = ["assets"]
