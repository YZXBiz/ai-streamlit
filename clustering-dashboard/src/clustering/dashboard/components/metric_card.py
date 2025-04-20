"""Metric card component for displaying key metrics in the dashboard."""

from typing import Any, Optional

import streamlit as st


def metric_card(
    title: str,
    value: Any,
    delta: Optional[float] = None,
    prefix: str = "",
    suffix: str = "",
    help_text: Optional[str] = None,
) -> None:
    """Display a metric in a stylized card format.

    Args:
        title: The title of the metric
        value: The value to display
        delta: Optional delta value to show a trend
        prefix: Prefix to display before the value (e.g., "$")
        suffix: Suffix to display after the value (e.g., "%")
        help_text: Optional help text to display on hover
    """
    formatted_value = f"{prefix}{value}{suffix}"

    if help_text:
        st.metric(
            label=title,
            value=formatted_value,
            delta=delta,
            help=help_text,
        )
    else:
        st.metric(
            label=title,
            value=formatted_value,
            delta=delta,
        )


def metric_row(metrics: list[dict[str, Any]], num_columns: int = 4) -> None:
    """Display a row of metric cards with equal spacing.

    Args:
        metrics: List of dictionaries with keys matching metric_card parameters
        num_columns: Number of columns to use for the layout
    """
    cols = st.columns(num_columns)

    for i, metric_data in enumerate(metrics):
        with cols[i % num_columns]:
            metric_card(**metric_data)
