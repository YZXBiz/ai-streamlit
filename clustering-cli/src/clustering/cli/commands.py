"""Command-line interface for the clustering project."""

import builtins
import json
import os
from typing import Any

import click
import pandas as pd
import yaml
from rich import print as rprint
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
def cli():
    """Clustering project CLI."""
    pass


@cli.command()
@click.argument("job_name", required=True)
@click.option("--env", default="dev", help="Environment to run in.")
@click.option("--config", type=click.Path(exists=True), help="Path to configuration file (JSON or YAML).")
@click.option("--param", multiple=True, help="Additional parameters in key=value format.")
def run(job_name: str, env: str, config: str | None = None, param: list[str] | None = None):
    """Run a pipeline job.

    Args:
        job_name: Name of the job to run
        env: Environment to run in (dev, test, prod)
        config: Optional path to configuration file (JSON or YAML)
        param: Optional additional parameters
    """
    rprint(f"Running job [bold]{job_name}[/bold] in [green]{env}[/green] environment")

    # Process parameters
    params = {}
    if param:
        for p in param:
            if "=" in p:
                key, value = p.split("=", 1)
                params[key] = value

    # Process config file
    if config:
        config_data = load_config_file(config)
        if config_data and "job" in config_data and "params" in config_data["job"]:
            params.update(config_data["job"]["params"])

    try:
        # Run the job
        result = run_job(job_name, env=env, params=params)
        rprint(
            f"[green]Job completed successfully![/green] Job ID: {result.get('job_id', 'unknown')}"
        )
        return 0
    except ImportError:
        rprint("[red]Error: Pipeline package not found. Make sure it's installed.[/red]")
        return 1  # Don't exit for tests
    except Exception as e:
        rprint(f"[red]Error running job: {str(e)}[/red]")
        return 1  # Don't exit for tests


@cli.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Path to configuration file to validate (JSON or YAML)."
)
@click.option("--data", type=click.Path(exists=True), help="Path to data file to validate.")
@click.option("--schema", help="Schema name for data validation.")
def validate(config: str | None = None, data: str | None = None, schema: str | None = None):
    """Validate configuration or data files.

    Args:
        config: Path to configuration file to validate (JSON or YAML)
        data: Path to data file to validate
        schema: Schema name for data validation
    """
    if not config and not data:
        rprint("[yellow]Please specify either --config or --data to validate.[/yellow]")
        return 1  # Don't exit for tests

    success = True

    if config:
        try:
            result = validate_config(config)
            if result["valid"]:
                rprint(f"[green]{result['message']}[/green]")
            else:
                rprint(f"[red]{result['message']}[/red]")
                if "errors" in result:
                    for error in result["errors"]:
                        rprint(f"  - {error}")
                success = False
        except Exception as e:
            rprint(f"[red]Error validating config: {str(e)}[/red]")
            return 1  # Don't exit for tests

    if data:
        try:
            result = validate_data(data, schema=schema)
            if result["valid"]:
                rprint(f"[green]{result['message']}[/green]")
                rprint(
                    f"Data file contains {result['rows']} rows with columns: {', '.join(result['columns'])}"
                )
            else:
                rprint(f"[red]{result['message']}[/red]")
                if "errors" in result:
                    for error in result["errors"]:
                        rprint(f"  - {error}")
                success = False
        except Exception as e:
            rprint(f"[red]Error validating data: {str(e)}[/red]")
            return 1  # Don't exit for tests

    return 0 if success else 1


@cli.command()
@click.argument("job_id", required=True)
@click.option("--output", type=click.Path(), help="Output file path.")
@click.option(
    "--format", type=click.Choice(["csv", "json", "excel"]), default="csv", help="Output format."
)
@click.option("--filter", help="Filter expression for the results.")
def export(job_id: str, output: str | None = None, format: str = "csv", filter: str | None = None):
    """Export job results to a file.

    Args:
        job_id: ID of the job to export results from
        output: Path to save the results
        format: Output format (csv, json, excel)
        filter: Optional filter expression
    """
    try:
        result = export_results(job_id, output_path=output, format=format, filter_expr=filter)
        if result["success"]:
            rprint(f"[green]{result['message']}[/green]")
            return 0
        else:
            rprint(f"[red]{result['message']}[/red]")
            return 1  # Don't exit for tests
    except Exception as e:
        rprint(f"[red]Error exporting results: {str(e)}[/red]")
        return 1  # Don't exit for tests


@cli.command()
@click.argument("job_id", required=False)
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def status(job_id: str | None = None, output: str = "table"):
    """Check status of a job or all jobs.

    Args:
        job_id: Optional job ID to check status for
        output: Output format (table or json)
    """
    try:
        if job_id:
            result = get_job_status(job_id)
            if output == "json":
                console.print_json(json.dumps(result))
            else:
                # Create a table
                table = Table(title=f"Status for Job: {job_id}")
                table.add_column("Field", style="cyan")
                table.add_column("Value", style="green")

                for key, value in result.items():
                    table.add_row(key, str(value))

                console.print(table)
        else:
            # Get status of all jobs
            results = get_all_jobs_status()
            if output == "json":
                console.print_json(json.dumps(results))
            else:
                # Create a table
                table = Table(title="All Jobs Status")
                table.add_column("Job ID", style="cyan")
                table.add_column("Status", style="green")
                table.add_column("Start Time", style="magenta")
                table.add_column("End Time", style="magenta")

                for job in results:
                    table.add_row(
                        job["job_id"],
                        job["status"],
                        job.get("start_time", "N/A"),
                        job.get("end_time", "N/A"),
                    )

                console.print(table)
        return 0
    except Exception as e:
        rprint(f"[red]Error getting status: {str(e)}[/red]")
        return 1  # Don't exit for tests


@cli.command()
@click.option("--type", help="Filter jobs by type.")
@click.option("--status", help="Filter jobs by status.")
@click.option("--limit", type=int, default=10, help="Maximum number of jobs to list.")
@click.option(
    "--output", type=click.Choice(["table", "json"]), default="table", help="Output format."
)
def list(
    type: str | None = None, status: str | None = None, limit: int = 10, output: str = "table"
):
    """List available jobs or job runs.

    Args:
        type: Filter jobs by type
        status: Filter jobs by status
        limit: Maximum number of jobs to list
        output: Output format (table or json)
    """
    try:
        results = list_jobs(job_type=type, status=status, limit=limit)

        if output == "json":
            console.print_json(json.dumps(results))
        else:
            # Create a table
            table = Table(title="Jobs List")
            table.add_column("Job ID", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Status", style="magenta")
            table.add_column("Created", style="blue")

            for job in results:
                table.add_row(
                    job["job_id"],
                    job.get("type", "N/A"),
                    job.get("status", "N/A"),
                    job.get("created_at", "N/A"),
                )

            console.print(table)
        return 0
    except Exception as e:
        rprint(f"[red]Error listing jobs: {str(e)}[/red]")
        return 1  # Don't exit for tests


@cli.command()
@click.option("--host", default="localhost", help="Host to bind to.")
@click.option("--port", default=3000, type=int, help="Port to bind to.")
@click.option("--env", default="dev", help="Environment to run in.")
def dashboard(host: str, port: int, env: str):
    """Start the Dagster dashboard UI.

    Args:
        host: Host to bind to
        port: Port to bind to
        env: Environment to run in (dev, test, prod)
    """
    rprint(f"Starting dashboard on {host}:{port} in [green]{env}[/green] environment")

    try:
        # Import pipeline modules at runtime to avoid circular imports
        from clustering.pipeline.app import run_app

        # Run the web app
        run_app(host=host, port=port, env=env)
        return 0
    except ImportError:
        rprint("[red]Error: Pipeline package not found. Make sure it's installed.[/red]")
        return 1  # Don't exit for tests
    except Exception as e:
        rprint(f"[red]Error starting dashboard: {str(e)}[/red]")
        return 1  # Don't exit for tests


@cli.command()
@click.option("--output", type=click.Path(), help="Output file path for the config template.")
@click.option("--type", type=click.Choice(["job", "pipeline", "full"]), default="full", help="Type of template to create.")
def create(output: str | None = None, type: str = "full"):
    """Create a template YAML configuration file.
    
    Args:
        output: Path to save the configuration template. If not provided, will prompt for input.
        type: Type of configuration template to create (job, pipeline, or full).
    """
    if not output:
        output = click.prompt("Enter path for the configuration file", default="config.yaml")
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output)) if os.path.dirname(output) else ".", exist_ok=True)
    
    # Create the template based on the type
    template = create_config_template(type)
    
    try:
        with open(output, "w") as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)
        
        rprint(f"[green]Created configuration template at: {output}[/green]")
        rprint("[yellow]Please edit this file to customize your configuration.[/yellow]")
        return 0
    except Exception as e:
        rprint(f"[red]Error creating template: {str(e)}[/red]")
        return 1  # Don't exit for tests


# Helper functions for the CLI commands
def run_job(job_name: str, env: str = "dev", params: dict[str, Any] = None) -> dict[str, Any]:
    """Run a job with the specified name and parameters.

    This is a mock implementation for testing. In a real implementation,
    this would call the appropriate Dagster job.

    Args:
        job_name: Name of the job to run
        env: Environment to run in
        params: Job parameters

    Returns:
        Dictionary with job execution results
    """
    # This is a mock implementation
    if "invalid" in job_name.lower():
        raise ValueError("Invalid job configuration")
    return {"status": "success", "job_id": f"{job_name}-123"}


def load_config_file(config_path: str) -> dict[str, Any] | None:
    """Load a configuration file in either JSON or YAML format.

    Args:
        config_path: Path to the configuration file

    Returns:
        Loaded configuration or None if loading fails
    """
    try:
        ext = os.path.splitext(config_path)[1].lower()
        with open(config_path, "r") as f:
            if ext == ".json":
                return json.load(f)
            elif ext in (".yaml", ".yml"):
                return yaml.safe_load(f)
            else:
                # Try to detect format based on content
                content = f.read()
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    try:
                        return yaml.safe_load(content)
                    except yaml.YAMLError:
                        rprint("[yellow]Warning: Could not determine file format. Attempting to parse as JSON.[/yellow]")
                        return json.loads(content)  # Will raise exception if not JSON
    except Exception as e:
        rprint(f"[red]Error loading config file: {str(e)}[/red]")
        return None


def validate_config(config_path: str) -> dict[str, Any]:
    """Validate a configuration file.

    Args:
        config_path: Path to the configuration file (JSON or YAML)

    Returns:
        Validation result
    """
    # Load the configuration file
    try:
        config = load_config_file(config_path)
        if config is None:
            return {
                "valid": False,
                "message": "Failed to load configuration file",
                "errors": ["Could not parse file format"],
            }

        # Simple validation check
        if "job" not in config:
            return {
                "valid": False,
                "message": "Invalid config format",
                "errors": ["Missing 'job' section"],
            }

        if "params" not in config["job"]:
            return {
                "valid": False,
                "message": "Invalid config format",
                "errors": ["Missing 'params' section"],
            }

        return {"valid": True, "message": "Config is valid"}
    except json.JSONDecodeError:
        return {
            "valid": False,
            "message": "Invalid JSON format",
            "errors": ["File is not valid JSON"],
        }
    except yaml.YAMLError:
        return {
            "valid": False,
            "message": "Invalid YAML format",
            "errors": ["File is not valid YAML"],
        }
    except Exception as e:
        return {"valid": False, "message": f"Validation error: {str(e)}", "errors": [str(e)]}


def validate_data(data_path: str, schema: str | None = None) -> dict[str, Any]:
    """Validate a data file.

    Args:
        data_path: Path to the data file
        schema: Optional schema name for validation

    Returns:
        Validation result
    """
    # This is a mock implementation
    try:
        # Read the data file
        df = pd.read_csv(data_path)

        # Return basic info
        return {
            "valid": True,
            "message": "Data is valid",
            "rows": len(df),
            "columns": df.columns.tolist(),
        }
    except Exception as e:
        return {"valid": False, "message": f"Data validation error: {str(e)}", "errors": [str(e)]}


def export_results(
    job_id: str,
    output_path: str | None = None,
    format: str = "csv",
    filter_expr: str | None = None,
) -> dict[str, Any]:
    """Export job results to a file.

    Args:
        job_id: ID of the job
        output_path: Path to save the results
        format: Output format (csv, json, excel)
        filter_expr: Optional filter expression

    Returns:
        Export result
    """
    # This is a mock implementation
    if not output_path:
        output_path = f"job_{job_id}_results.{format}"

    # Create a mock DataFrame
    df = pd.DataFrame(
        {
            "store_id": [f"STORE_{i}" for i in range(1, 11)],
            "cluster": [i % 3 + 1 for i in range(10)],
            "score": [0.8 + i / 100 for i in range(10)],
        }
    )

    # Apply filter if provided
    filtered = False
    if filter_expr:
        # Simple parsing for demo (not a real implementation)
        if "cluster" in filter_expr and "=" in filter_expr:
            _, value = filter_expr.split("=")
            try:
                cluster_value = int(value.strip())
                df = df[df["cluster"] == cluster_value]
                filtered = True
            except ValueError:
                pass

    # Write to file based on format
    try:
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "json":
            df.to_json(output_path, orient="records", indent=2)
        elif format == "excel":
            df.to_excel(output_path, index=False)

        return {
            "success": True,
            "message": f"Exported {len(df)} {'filtered ' if filtered else ''}results to {output_path}",
            "rows": len(df),
            "path": output_path,
        }
    except Exception as e:
        return {"success": False, "message": f"Error exporting results: {str(e)}"}


def get_job_status(job_id: str) -> dict[str, Any]:
    """Get status information for a specific job.

    Args:
        job_id: ID of the job

    Returns:
        Job status information
    """
    # This is a mock implementation
    return {
        "job_id": job_id,
        "status": "COMPLETED",
        "start_time": "2023-01-01T12:00:00",
        "end_time": "2023-01-01T12:15:00",
        "duration": "15m 0s",
    }


def get_all_jobs_status() -> builtins.list[dict[str, Any]]:
    """Get status information for all jobs.

    Returns:
        List of job status information
    """
    # This is a mock implementation
    return [
        {
            "job_id": "job-001",
            "status": "completed",
            "start_time": "2023-01-01T10:00:00",
            "end_time": "2023-01-01T10:15:45",
        },
        {
            "job_id": "job-002",
            "status": "running",
            "start_time": "2023-01-01T11:30:00",
            "end_time": None,
        },
        {
            "job_id": "job-003",
            "status": "failed",
            "start_time": "2023-01-01T12:45:00",
            "end_time": "2023-01-01T12:50:22",
        },
    ]


def list_jobs(
    job_type: str | None = None, status: str | None = None, limit: int = 10
) -> builtins.list[dict[str, Any]]:
    """List jobs with optional filtering.

    Args:
        job_type: Optional job type filter
        status: Optional status filter
        limit: Maximum number of jobs to return

    Returns:
        List of jobs
    """
    # This is a mock implementation
    jobs = [
        {
            "job_id": f"job-{i}",
            "type": "clustering" if i % 3 == 0 else "preprocessing" if i % 3 == 1 else "validation",
            "status": "COMPLETED"
            if i % 4 == 0
            else "RUNNING"
            if i % 4 == 1
            else "FAILED"
            if i % 4 == 2
            else "PENDING",
            "created_at": f"2023-01-{1 + i % 10}T10:00:00",
        }
        for i in range(1, 6)  # Only return 5 jobs by default
    ]

    # Apply filters
    if job_type:
        jobs = [j for j in jobs if j["type"] == job_type]

    if status:
        jobs = [j for j in jobs if j["status"] == status]

    # Apply limit
    jobs = jobs[:limit]

    return jobs


def create_config_template(template_type: str = "full") -> dict[str, Any]:
    """Create a configuration template based on the specified type.
    
    Args:
        template_type: Type of template to create (job, pipeline, or full)
        
    Returns:
        Configuration template as a dictionary
    """
    # Base job template
    job_template = {
        "job": {
            "name": "clustering-pipeline",
            "params": {
                "input_path": "<PATH_TO_INPUT_DATA>",
                "output_path": "<PATH_TO_OUTPUT_DATA>",
                "num_clusters": 5,
                "algorithm": "kmeans",
                "features": [
                    "sales_amount",
                    "transaction_count",
                    "avg_basket_size"
                ],
                "normalize": True,
                "random_seed": 42
            }
        }
    }
    
    # Pipeline configuration template
    pipeline_template = {
        "alerts": {
            "enabled": True,
            "threshold": "WARNING"
        },
        "logger": {
            "level": "DEBUG",
            "sink": "logs/dagster_dev.log"
        },
        "paths": {
            "base_data_dir": "<PATH_TO_DATA_DIR>",
            "internal_data_dir": "<PATH_TO_INTERNAL_DATA_DIR>",
            "external_data_dir": "<PATH_TO_EXTERNAL_DATA_DIR>",
            "merging_data_dir": "<PATH_TO_MERGING_DATA_DIR>"
        },
        "job_params": {
            "ignore_features": ["STORE_NBR"],
            "normalize": True,
            "norm_method": "robust",
            "imputation_type": "simple",
            "numeric_imputation": "mean",
            "categorical_imputation": "mode",
            "outlier_detection": True,
            "outliers_method": "iforest",
            "outlier_threshold": 0.05,
            "pca_active": True,
            "pca_method": "linear",
            "pca_components": 0.8,
            "min_clusters": 2,
            "max_clusters": 10,
            "metrics": ["silhouette", "calinski_harabasz", "davies_bouldin"],
            "algorithm": "kmeans",
            "session_id": 42
        }
    }
    
    # Full template includes data readers and writers
    full_template = pipeline_template.copy()
    full_template["readers"] = {
        "sales_data": {
            "kind": "CSVReader",
            "config": {
                "path": "<PATH_TO_SALES_DATA>",
                "ignore_errors": True,
                "delimiter": ",",
                "null_values": ["", "NA", "N/A", "None", "null"],
                "encoding": "utf8",
                "try_parse_dates": False,
                "comment_char": None
            }
        },
        "external_data": {
            "kind": "CSVReader",
            "config": {
                "path": "<PATH_TO_EXTERNAL_DATA>",
                "ignore_errors": True,
                "delimiter": ",",
                "null_values": ["", "NA", "N/A", "None", "null"],
                "encoding": "utf8",
                "try_parse_dates": False,
                "comment_char": None
            }
        }
    }
    
    full_template["writers"] = {
        "feature_engineering_output": {
            "kind": "PickleWriter",
            "config": {
                "path": "<PATH_TO_ENGINEERED_FEATURES>"
            }
        },
        "model_output": {
            "kind": "PickleWriter",
            "config": {
                "path": "<PATH_TO_MODEL_OUTPUT>"
            }
        },
        "cluster_assignments": {
            "kind": "PickleWriter",
            "config": {
                "path": "<PATH_TO_CLUSTER_ASSIGNMENTS>"
            }
        },
        "merged_clusters_output": {
            "kind": "CSVWriter",
            "config": {
                "path": "<PATH_TO_MERGED_CLUSTERS>",
                "index": False
            }
        }
    }
    
    if template_type == "job":
        return job_template
    elif template_type == "pipeline":
        return pipeline_template
    else:  # full
        return full_template


# Command aliases for Click
run_command = run
validate_command = validate
export_command = export
status_command = status
list_command = list
create_command = create


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
