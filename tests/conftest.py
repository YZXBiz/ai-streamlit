"""Pytest configuration for tests."""

import os
import sys
from collections.abc import AsyncGenerator, Generator
from datetime import timedelta
from pathlib import Path
from typing import Dict

# Add the backend directory to the Python path
backend_dir = str(Path(__file__).parent.parent / "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Force the env value to be a clean integer
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "1440"

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Set environment file for testing
os.environ["ENV_FILE"] = ".env.test"

# ... existing code ...
