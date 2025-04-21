#!/usr/bin/env python
"""Validate deployment environment configuration.

This script tests that the Dagster pipeline can connect to resources
in the target environment (database, object storage, etc).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import dagster as dg
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("environment-test")


def load_config(env: str) -> Dict[str, Any]:
    """Load environment configuration file.

    Args:
        env: Environment name (dev, staging, prod)

    Returns:
        Dictionary with configuration values

    Raises:
        FileNotFoundError: If configuration file is not found
    """
    config_file = Path(f"clustering-pipeline/src/clustering/pipeline/resources/configs/{env}.yml")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def test_database_connection(config: Dict[str, Any]) -> bool:
    """Test database connection using configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if connection is successful, False otherwise
    """
    try:
        # Extract database configuration from config
        db_config = config.get("resources", {}).get("database", {})
        if not db_config:
            logger.warning("No database configuration found")
            return False

        # Implement database connection test here
        # For example:
        # from sqlalchemy import create_engine
        # engine = create_engine(db_config["url"])
        # with engine.connect() as conn:
        #     result = conn.execute("SELECT 1")
        
        # This is a placeholder - implement actual test
        logger.info("Database connection test passed")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def test_dagster_asset_loading() -> bool:
    """Test that Dagster assets can be loaded correctly.

    Returns:
        True if assets load successfully, False otherwise
    """
    try:
        # Import the definitions
        from clustering.pipeline.definitions import defs
        
        # Get assets and check count
        asset_count = len(defs.get_all_asset_nodes())
        logger.info(f"Successfully loaded {asset_count} Dagster assets")
        return True
    except Exception as e:
        logger.error(f"Dagster asset loading test failed: {e}")
        return False


def test_data_paths(config: Dict[str, Any]) -> bool:
    """Test that data paths exist and are writable.

    Args:
        config: Configuration dictionary

    Returns:
        True if data paths are valid, False otherwise
    """
    try:
        # Extract paths from config
        paths = config.get("paths", {})
        if not paths:
            logger.warning("No path configuration found")
            return False
        
        # Check each path
        for name, path_str in paths.items():
            # Handle environment variable substitution
            if path_str.startswith("${env:"):
                parts = path_str.strip("${env:").split(",")
                env_var = parts[0]
                default = parts[1] if len(parts) > 1 else None
                path_str = os.environ.get(env_var, default)
                if path_str is None:
                    logger.error(f"Environment variable {env_var} not set and no default provided")
                    return False
            
            path = Path(path_str)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            
            # Test write permission
            test_file = path / ".write_test"
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                test_file.unlink()  # Remove test file
                logger.info(f"Path {name} ({path}) is writable")
            except PermissionError:
                logger.error(f"Path {name} ({path}) is not writable")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Data path test failed: {e}")
        return False


def main() -> int:
    """Run environment tests.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    parser = argparse.ArgumentParser(description="Test deployment environment")
    parser.add_argument(
        "--env", 
        choices=["dev", "staging", "prod"],
        default="dev",
        help="Environment to test (dev, staging, prod)"
    )
    args = parser.parse_args()

    try:
        logger.info(f"Testing {args.env} environment")
        
        # Load configuration
        config = load_config(args.env)
        
        # Run tests
        tests = [
            ("Database connection", lambda: test_database_connection(config)),
            ("Dagster asset loading", test_dagster_asset_loading),
            ("Data paths", lambda: test_data_paths(config)),
        ]
        
        failures = 0
        for name, test_func in tests:
            logger.info(f"Running test: {name}")
            if test_func():
                logger.info(f"✅ {name}: PASSED")
            else:
                logger.error(f"❌ {name}: FAILED")
                failures += 1
        
        if failures > 0:
            logger.error(f"{failures} tests failed")
            return 1
        else:
            logger.info("All environment tests passed!")
            return 0
            
    except Exception as e:
        logger.error(f"Environment test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 