"""Time-based partitions for the clustering pipeline."""

import datetime

from dagster import (
    DailyPartitionsDefinition,
    MonthlyPartitionsDefinition,
    WeeklyPartitionsDefinition,
)


def get_daily_partitions(
    start_date: str | None = None,
    end_offset: int = 1,
    timezone: str = "UTC",
) -> DailyPartitionsDefinition:
    """Get daily partitions definition.

    Args:
        start_date: Start date in YYYY-MM-DD format (defaults to 30 days ago)
        end_offset: Offset from current date (defaults to 1 day ago)
        timezone: Timezone for partitions

    Returns:
        DailyPartitionsDefinition
    """
    if not start_date:
        # Default to 30 days ago
        start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")

    return DailyPartitionsDefinition(
        start_date=start_date,
        end_offset=end_offset,
        timezone=timezone,
        fmt="%Y-%m-%d",
    )


def get_weekly_partitions(
    start_date: str | None = None,
    end_offset: int = 1,
    timezone: str = "UTC",
    start_day_of_week: int = 0,  # Monday
) -> WeeklyPartitionsDefinition:
    """Get weekly partitions definition.

    Args:
        start_date: Start date in YYYY-MM-DD format (defaults to 8 weeks ago)
        end_offset: Number of weeks from current date for end
        timezone: Timezone for partitions
        start_day_of_week: Starting day of week (0 = Monday, 6 = Sunday)

    Returns:
        WeeklyPartitionsDefinition
    """
    if not start_date:
        # Default to 8 weeks ago, aligned to the start day of week
        today = datetime.datetime.now()
        days_since_start = (today.weekday() - start_day_of_week) % 7
        start_of_week = today - datetime.timedelta(days=days_since_start)
        start_date = (start_of_week - datetime.timedelta(weeks=8)).strftime("%Y-%m-%d")

    return WeeklyPartitionsDefinition(
        start_date=start_date,
        end_offset=end_offset,
        timezone=timezone,
        fmt="%Y-%m-%d",
        start_day_of_week=start_day_of_week,
    )


def get_monthly_partitions(
    start_date: str | None = None,
    end_offset: int = 1,
    timezone: str = "UTC",
) -> MonthlyPartitionsDefinition:
    """Get monthly partitions definition.

    Args:
        start_date: Start date in YYYY-MM-DD format (defaults to 6 months ago)
        end_offset: Number of months from current date for end
        timezone: Timezone for partitions

    Returns:
        MonthlyPartitionsDefinition
    """
    if not start_date:
        # Default to 6 months ago
        today = datetime.datetime.now()
        months_ago = today.month - 6
        year_offset = 0
        while months_ago <= 0:
            months_ago += 12
            year_offset -= 1
        start_date = f"{today.year + year_offset}-{months_ago:02d}-01"

    return MonthlyPartitionsDefinition(
        start_date=start_date,
        end_offset=end_offset,
        timezone=timezone,
        fmt="%Y-%m-%d",
    )


# Common partition instances
daily_partitions = get_daily_partitions()
weekly_partitions = get_weekly_partitions()
monthly_partitions = get_monthly_partitions()
