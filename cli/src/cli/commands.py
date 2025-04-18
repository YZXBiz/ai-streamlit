"""Command-line interface for the clustering project."""

import click
from rich import print as rprint


@click.group()
def cli():
    """Clustering project CLI."""
    pass


@cli.command()
@click.argument("job_name", required=True)
@click.option("--env", default="dev", help="Environment to run in.")
def run(job_name: str, env: str):
    """Run a pipeline job.
    
    Args:
        job_name: Name of the job to run
        env: Environment to run in (dev, test, prod)
    """
    rprint(f"Running job [bold]{job_name}[/bold] in [green]{env}[/green] environment")
    
    try:
        # Import pipeline modules at runtime to avoid circular imports
        from pipeline.app import run_job
        
        # Run the job
        run_job(job_name=job_name, env=env)
        rprint("[green]Job completed successfully![/green]")
    except ImportError:
        rprint("[red]Error: Pipeline package not found. Make sure it's installed.[/red]")
    except Exception as e:
        rprint(f"[red]Error running job: {str(e)}[/red]")


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
        from pipeline.app import run_app
        
        # Run the web app
        run_app(host=host, port=port, env=env)
    except ImportError:
        rprint("[red]Error: Pipeline package not found. Make sure it's installed.[/red]")
    except Exception as e:
        rprint(f"[red]Error starting dashboard: {str(e)}[/red]")


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main() 