"""
PandasAI Chat Application - Backend API

This package implements a FastAPI backend for the PandasAI Chat application,
following clean architecture principles.
"""

__version__ = "0.1.0"

from backend.app.services.analyzer_service import AnalyzerService
from backend.app.services.dataframe_service import DataFrameService as DataFrameService


def create_analyzer() -> AnalyzerService:
    """
    Create a new AnalyzerService instance.

    Returns:
        A new AnalyzerService instance
    """
    return AnalyzerService()
