"""Schedules for the clustering pipeline."""

import dagster as dg


@dg.schedule(
    cron_schedule="0 0 * * *",  # Daily at midnight
    job_name="internal_clustering_job",
    execution_timezone="UTC",
)
def daily_internal_clustering_schedule(context: dg.ScheduleEvaluationContext) -> dg.RunRequest:
    """Schedule for daily execution of the internal clustering pipeline.

    Args:
        context: Schedule evaluation context

    Returns:
        RunRequest for the job
    """
    return dg.RunRequest(
        run_key=context.scheduled_execution_time.strftime("%Y-%m-%d-%H-%M"),
        tags={"schedule": "daily_internal_clustering"},
    )


@dg.schedule(
    cron_schedule="0 0 * * 1",  # Weekly on Monday
    job_name="external_clustering_job",
    execution_timezone="UTC",
)
def weekly_external_clustering_schedule(context: dg.ScheduleEvaluationContext) -> dg.RunRequest:
    """Schedule for weekly execution of the external clustering pipeline.

    Args:
        context: Schedule evaluation context

    Returns:
        RunRequest for the job
    """
    return dg.RunRequest(
        run_key=context.scheduled_execution_time.strftime("%Y-%m-%d-%H-%M"),
        tags={"schedule": "weekly_external_clustering"},
    )


@dg.schedule(
    cron_schedule="0 0 1 * *",  # Monthly on the 1st
    job_name="full_pipeline_job",
    execution_timezone="UTC",
)
def monthly_full_pipeline_schedule(context: dg.ScheduleEvaluationContext) -> dg.RunRequest:
    """Schedule for monthly execution of the full pipeline.

    Args:
        context: Schedule evaluation context

    Returns:
        RunRequest for the job
    """
    return dg.RunRequest(
        run_key=context.scheduled_execution_time.strftime("%Y-%m-%d-%H-%M"),
        tags={"schedule": "monthly_full_pipeline"},
    )


schedules = [
    daily_internal_clustering_schedule,
    weekly_external_clustering_schedule,
    monthly_full_pipeline_schedule,
]
