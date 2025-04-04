#!/usr/bin/env python3
"""
Data pipeline monitoring script.

This script monitors the data flow through the pipeline,
tracking metrics and validating data quality at each step.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any

import polars as pl
from rich.console import Console
from rich.table import Table

# Add the project directory to Python path if needed
sys.path.append(".")

from src.clustering.core.sql_engine import DuckDB


class DataMonitor:
    """Monitor data flow through the pipeline and track metrics."""

    def __init__(self, output_dir: str, report_path: str | None = None) -> None:
        """Initialize the DataMonitor.

        Args:
            output_dir: Directory containing pipeline output files
            report_path: Path to save the monitoring report (defaults to monitor_report.json)
        """
        self.output_dir = output_dir
        self.report_path = report_path or os.path.join(
            output_dir, f"monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.metrics: dict[str, dict[str, Any]] = {}
        self.db = DuckDB()
        self.console = Console()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.db.close()
        if self.metrics:
            self.save_report()

    def analyze_dataframe(
        self, name: str, df: pl.DataFrame, check_nulls: bool = True, check_outliers: bool = False
    ) -> None:
        """Analyze a DataFrame and record metrics.

        Args:
            name: Name of the DataFrame
            df: DataFrame to analyze
            check_nulls: Whether to check for null values
            check_outliers: Whether to check for outliers in numeric columns
        """
        metrics = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns,
            "timestamp": datetime.now().isoformat(),
        }

        # Column-level metrics
        column_metrics = {}
        for col in df.columns:
            col_metrics = {}

            # Data type
            col_metrics["dtype"] = str(df[col].dtype)

            # Check for nulls if requested
            if check_nulls:
                null_count = df[col].null_count()
                col_metrics["null_count"] = null_count
                col_metrics["null_percentage"] = (
                    round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0
                )

            # Basic statistics for numeric columns
            try:
                if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                    describe = df[col].describe()
                    col_metrics["min"] = describe.filter(pl.col("statistic") == "min")["value"][0]
                    col_metrics["max"] = describe.filter(pl.col("statistic") == "max")["value"][0]
                    col_metrics["mean"] = describe.filter(pl.col("statistic") == "mean")["value"][0]
                    col_metrics["std"] = describe.filter(pl.col("statistic") == "std")["value"][0]

                    # Check for outliers if requested
                    if check_outliers:
                        q1 = df[col].quantile(0.25)
                        q3 = df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        outliers = df.filter(
                            (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
                        )
                        col_metrics["outlier_count"] = len(outliers)
                        col_metrics["outlier_percentage"] = (
                            round((len(outliers) / len(df)) * 100, 2) if len(df) > 0 else 0
                        )
            except Exception as e:
                col_metrics["stats_error"] = str(e)

            # For categorical columns, count unique values
            if df[col].dtype in [pl.Categorical, pl.Utf8, pl.String]:
                try:
                    unique_count = df[col].n_unique()
                    col_metrics["unique_values"] = unique_count
                    col_metrics["unique_percentage"] = (
                        round((unique_count / len(df)) * 100, 2) if len(df) > 0 else 0
                    )

                    # If fewer than 10 unique values, show the value counts
                    if 0 < unique_count < 10:
                        value_counts = df[col].value_counts()
                        col_metrics["value_counts"] = {
                            str(row[col]): row["counts"]
                            for row in value_counts.iter_rows(named=True)
                        }
                except Exception as e:
                    col_metrics["unique_error"] = str(e)

            column_metrics[col] = col_metrics

        metrics["column_metrics"] = column_metrics
        self.metrics[name] = metrics

    def scan_output_dir(self) -> list[str]:
        """Scan the output directory for Parquet files.

        Returns:
            List of Parquet file paths found
        """
        parquet_files = []
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
        return parquet_files

    def monitor_all_outputs(self, check_nulls: bool = True, check_outliers: bool = False) -> None:
        """Monitor all Parquet files in the output directory.

        Args:
            check_nulls: Whether to check for null values
            check_outliers: Whether to check for outliers in numeric columns
        """
        parquet_files = self.scan_output_dir()
        self.console.print(
            f"Found [bold green]{len(parquet_files)}[/] Parquet files in {self.output_dir}"
        )

        for file_path in parquet_files:
            try:
                # Extract file name without extension for use as metric name
                file_name = os.path.basename(file_path).replace(".parquet", "")
                self.console.print(f"Analyzing [bold blue]{file_name}[/]...")

                # Read the Parquet file
                df = pl.read_parquet(file_path)

                # Analyze the DataFrame
                self.analyze_dataframe(file_name, df, check_nulls, check_outliers)

                self.console.print(
                    f"✅ Successfully analyzed [bold green]{file_name}[/] ({len(df)} rows)"
                )
            except Exception as e:
                self.console.print(f"❌ Error analyzing {file_path}: {str(e)}", style="bold red")

    def generate_summary(self) -> None:
        """Generate a summary of the monitored data."""
        if not self.metrics:
            self.console.print("No metrics data available", style="bold red")
            return

        # Create a summary table
        table = Table(title="Pipeline Data Summary")
        table.add_column("Dataset", style="cyan")
        table.add_column("Rows", justify="right", style="green")
        table.add_column("Columns", justify="right")
        table.add_column("Null %", justify="right")
        table.add_column("Issues", style="red")

        for name, metrics in self.metrics.items():
            row_count = metrics["row_count"]
            col_count = metrics["column_count"]

            # Calculate average null percentage across columns
            null_percentages = [
                m.get("null_percentage", 0)
                for m in metrics.get("column_metrics", {}).values()
                if "null_percentage" in m
            ]
            avg_null_percentage = (
                round(sum(null_percentages) / len(null_percentages), 2) if null_percentages else 0
            )

            # Identify potential issues
            issues = []
            if row_count == 0:
                issues.append("Empty dataset")

            columns_with_high_nulls = [
                col
                for col, m in metrics.get("column_metrics", {}).items()
                if m.get("null_percentage", 0) > 50
            ]
            if columns_with_high_nulls:
                issues.append(f"{len(columns_with_high_nulls)} columns with >50% nulls")

            columns_with_outliers = [
                col
                for col, m in metrics.get("column_metrics", {}).items()
                if m.get("outlier_percentage", 0) > 5
            ]
            if columns_with_outliers:
                issues.append(f"{len(columns_with_outliers)} columns with >5% outliers")

            # Add row to the table
            table.add_row(
                name,
                str(row_count),
                str(col_count),
                str(avg_null_percentage) + "%",
                ", ".join(issues) if issues else "None",
            )

        self.console.print(table)

    def save_report(self) -> None:
        """Save the metrics report to JSON."""
        with open(self.report_path, "w") as f:
            json.dump(self.metrics, f, indent=2)
        self.console.print(f"Saved monitoring report to [bold]{self.report_path}[/]")


def monitor_recent_run():
    """Monitor the most recent pipeline run."""
    # Find the most recent output directory
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        print(f"Error: {outputs_dir} directory not found")
        return

    # Look for subdirectories that might contain pipeline outputs
    candidate_dirs = []
    for subdir in ["preprocessing", "test"]:
        full_dir = os.path.join(outputs_dir, subdir)
        if os.path.exists(full_dir):
            # Get all subdirectories sorted by modification time (newest first)
            run_dirs = [
                os.path.join(full_dir, d)
                for d in os.listdir(full_dir)
                if os.path.isdir(os.path.join(full_dir, d))
            ]
            run_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            candidate_dirs.extend(run_dirs)

    if not candidate_dirs:
        print("No recent runs found in outputs directory")
        return

    # Use the most recent directory
    latest_dir = candidate_dirs[0]
    console = Console()
    console.print(f"Monitoring data from most recent run: [bold blue]{latest_dir}[/]")

    with DataMonitor(latest_dir) as monitor:
        monitor.monitor_all_outputs(check_nulls=True, check_outliers=True)
        monitor.generate_summary()


if __name__ == "__main__":
    monitor_recent_run()
