#!/usr/bin/env python
"""
Pipeline validation script.

This script validates the pipeline configuration and data files to identify
potential issues before running the pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Setup path for imports
sys.path.append(".")


class ValidationResult:
    """Represents the result of a validation check."""

    def __init__(self, name: str, passed: bool, message: str, details: Optional[str] = None):
        """Initialize validation result.

        Args:
            name: Name of the validation check
            passed: Whether the check passed
            message: A brief message about the result
            details: Optional detailed message about the result
        """
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details

    def __str__(self) -> str:
        """Return string representation of the validation result."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        result = f"{status} | {self.name}: {self.message}"
        if self.details:
            result += f"\n   Details: {self.details}"
        return result


class PipelineValidator:
    """Validates the pipeline configuration and data files."""

    def __init__(self, base_dir: str = None):
        """Initialize the validator.

        Args:
            base_dir: Base directory of the project
        """
        self.base_dir = Path(base_dir or os.getcwd())
        self.results: List[ValidationResult] = []
        self.config_files: Dict[str, Dict] = {}
        self.env = "dev"  # Default environment

    def load_config_files(self) -> bool:
        """Load all configuration files.

        Returns:
            Whether the configuration files were loaded successfully
        """
        config_dir = self.base_dir / "src" / "clustering" / "dagster" / "resources" / "configs"

        # Check if the config directory exists
        if not config_dir.exists():
            self.results.append(
                ValidationResult("Config Directory Check", False, f"Config directory not found: {config_dir}")
            )
            return False

        # Load base config
        base_config_path = config_dir / "base.yml"
        if base_config_path.exists():
            with open(base_config_path, "r") as f:
                self.config_files["base"] = yaml.safe_load(f) or {}
        else:
            self.results.append(
                ValidationResult("Base Config Check", False, f"Base config file not found: {base_config_path}")
            )

        # Load environment config
        env_config_path = config_dir / f"{self.env}.yml"
        if env_config_path.exists():
            with open(env_config_path, "r") as f:
                self.config_files[self.env] = yaml.safe_load(f) or {}
        else:
            self.results.append(
                ValidationResult(
                    f"{self.env.capitalize()} Config Check",
                    False,
                    f"Environment config file not found: {env_config_path}",
                )
            )

        # Return whether we loaded any config files
        return len(self.config_files) > 0

    def get_config_value(self, path: str, default=None):
        """Get a value from the configuration.

        Args:
            path: Dot-separated path to the value
            default: Default value if not found

        Returns:
            The value from the configuration
        """
        parts = path.split(".")

        # Try environment-specific config first
        if self.env in self.config_files:
            value = self._get_nested_value(self.config_files[self.env], parts)
            if value is not None:
                return value

        # Fall back to base config
        if "base" in self.config_files:
            value = self._get_nested_value(self.config_files["base"], parts)
            if value is not None:
                return value

        # Return default if not found
        return default

    def _get_nested_value(self, data: Dict, keys: List[str]):
        """Get a nested value from a dictionary.

        Args:
            data: Dictionary to get the value from
            keys: List of keys to traverse

        Returns:
            The value at the given keys, or None if not found
        """
        current = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def check_data_files(self) -> None:
        """Check if data files referenced in the configuration exist."""
        # Check internal sales data file
        sales_path = self.get_config_value("readers.internal_sales.path", "data/raw/internal_sales.parquet")
        sales_file = self.base_dir / sales_path
        self.results.append(
            ValidationResult(
                "Internal Sales Data File Check",
                sales_file.exists(),
                "Internal sales data file exists" if sales_file.exists() else "Internal sales data file not found",
                f"Path: {sales_file}",
            )
        )

        # Check need state data file
        need_state_path = self.get_config_value("readers.internal_need_state.path", "data/raw/need_state.csv")
        need_state_file = self.base_dir / need_state_path
        self.results.append(
            ValidationResult(
                "Need State Data File Check",
                need_state_file.exists(),
                "Need state data file exists" if need_state_file.exists() else "Need state data file not found",
                f"Path: {need_state_file}",
            )
        )

        # Check external sales data file
        external_path = self.get_config_value("readers.external_sales.path", "data/raw/external_sales.parquet")
        external_file = self.base_dir / external_path
        self.results.append(
            ValidationResult(
                "External Sales Data File Check",
                external_file.exists(),
                "External sales data file exists" if external_file.exists() else "External sales data file not found",
                f"Path: {external_file}",
            )
        )

    def check_output_directories(self) -> None:
        """Check if output directories exist and are writable."""
        # Check outputs directory
        outputs_dir = self.base_dir / "outputs"
        outputs_exists = outputs_dir.exists()
        outputs_writable = outputs_exists and os.access(outputs_dir, os.W_OK)
        self.results.append(
            ValidationResult(
                "Outputs Directory Check",
                outputs_exists and outputs_writable,
                "Outputs directory exists and is writable"
                if outputs_exists and outputs_writable
                else "Outputs directory does not exist or is not writable",
                f"Path: {outputs_dir}",
            )
        )

        # Check logs directory
        logs_dir = self.base_dir / "logs"
        logs_exists = logs_dir.exists()
        logs_writable = logs_exists and os.access(logs_dir, os.W_OK)
        self.results.append(
            ValidationResult(
                "Logs Directory Check",
                logs_exists and logs_writable,
                "Logs directory exists and is writable"
                if logs_exists and logs_writable
                else "Logs directory does not exist or is not writable",
                f"Path: {logs_dir}",
            )
        )

    def check_dependencies(self) -> None:
        """Check if required dependencies are installed."""
        # Check for polars
        try:
            import polars

            polars_version = polars.__version__
            self.results.append(
                ValidationResult("Polars Dependency Check", True, f"Polars is installed (version {polars_version})")
            )
        except ImportError:
            self.results.append(
                ValidationResult(
                    "Polars Dependency Check", False, "Polars is not installed", "Run 'pip install polars' to install"
                )
            )

        # Check for dagster
        try:
            import dagster

            dagster_version = dagster.__version__
            self.results.append(
                ValidationResult("Dagster Dependency Check", True, f"Dagster is installed (version {dagster_version})")
            )
        except ImportError:
            self.results.append(
                ValidationResult(
                    "Dagster Dependency Check",
                    False,
                    "Dagster is not installed",
                    "Run 'pip install dagster' to install",
                )
            )

        # Check for dagster-duckdb
        try:
            import dagster_duckdb

            dagster_duckdb_version = dagster_duckdb.__version__
            self.results.append(
                ValidationResult(
                    "Dagster-DuckDB Dependency Check",
                    True,
                    f"Dagster-DuckDB is installed (version {dagster_duckdb_version})",
                )
            )
        except ImportError:
            self.results.append(
                ValidationResult(
                    "Dagster-DuckDB Dependency Check",
                    False,
                    "Dagster-DuckDB is not installed",
                    "Run 'pip install dagster-duckdb' to install",
                )
            )

    def check_source_files(self) -> None:
        """Check if key source files exist."""
        # Check key Python modules
        key_modules = [
            ("src/clustering/dagster/definitions.py", "Main Dagster definitions"),
            ("src/clustering/dagster/assets/preprocessing/internal.py", "Internal preprocessing assets"),
            ("src/clustering/dagster/resources/io_manager.py", "IO Manager"),
        ]

        for path, desc in key_modules:
            module_path = self.base_dir / path
            self.results.append(
                ValidationResult(
                    f"{desc} Check",
                    module_path.exists(),
                    f"{desc} file exists" if module_path.exists() else f"{desc} file not found",
                    f"Path: {module_path}",
                )
            )

    def validate(self, env: str = "dev") -> List[ValidationResult]:
        """Run all validation checks.

        Args:
            env: Environment to validate

        Returns:
            List of validation results
        """
        self.env = env
        self.results = []

        # Load configuration files
        if self.load_config_files():
            # Run checks
            self.check_data_files()
            self.check_output_directories()
            self.check_dependencies()
            self.check_source_files()

        return self.results

    def print_results(self) -> Tuple[int, int]:
        """Print validation results.

        Returns:
            Tuple of (passed_count, total_count)
        """
        print("\n=== Pipeline Validation Results ===\n")

        passed = 0
        for result in self.results:
            print(result)
            if result.passed:
                passed += 1

        total = len(self.results)
        print(f"\nSummary: {passed}/{total} checks passed ({passed / total * 100:.1f}%)")

        return passed, total


def main():
    """Run the validation script."""
    # Parse command line arguments
    env = "dev"
    if len(sys.argv) > 1:
        env = sys.argv[1]

    # Validate the pipeline
    validator = PipelineValidator()
    validator.validate(env)
    passed, total = validator.print_results()

    # Return a non-zero exit code if any checks failed
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
