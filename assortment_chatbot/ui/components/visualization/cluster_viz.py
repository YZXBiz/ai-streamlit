"""
Cluster visualization components for displaying clustering results.

This module provides visualization functions for displaying clustering results,
including distributions and feature analysis by cluster.
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def plot_cluster_distribution(df: pd.DataFrame, cluster_col: str) -> None:
    """Plot the distribution of items across clusters.

    Args:
        df: DataFrame containing cluster data
        cluster_col: Name of the column containing cluster assignments
    """
    if cluster_col not in df.columns:
        st.error(f"DataFrame must contain a '{cluster_col}' column")
        return

    # Count items per cluster
    cluster_counts = df[cluster_col].value_counts().sort_index()

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        [f"Cluster {c}" for c in cluster_counts.index],
        cluster_counts.values,
        color=plt.cm.tab10.colors[: len(cluster_counts)],
    )

    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    ax.set_title("Number of Items per Cluster", fontsize=16)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig)


def plot_feature_by_cluster(df: pd.DataFrame, feature_col: str, cluster_col: str) -> None:
    """Plot the distribution of a feature across clusters.

    Args:
        df: DataFrame containing cluster data
        feature_col: Column name of the feature to plot
        cluster_col: Name of the column containing cluster assignments
    """
    if cluster_col not in df.columns:
        st.error(f"DataFrame must contain a '{cluster_col}' column")
        return

    if feature_col not in df.columns:
        st.error(f"Feature column '{feature_col}' not found in DataFrame")
        return

    # Group by cluster and calculate statistics
    cluster_stats = df.groupby(cluster_col)[feature_col].agg(["mean", "std"]).reset_index()

    # Create a bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    x = [f"Cluster {c}" for c in cluster_stats[cluster_col]]
    y = cluster_stats["mean"]
    err = cluster_stats["std"]

    bars = ax.bar(
        x, y, yerr=err, capsize=10, color=plt.cm.tab10.colors[: len(cluster_stats)], alpha=0.7
    )

    # Add mean value labels on top of bars
    for bar, mean_val in zip(bars, y, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_title(f"Average {feature_col} by Cluster", fontsize=16)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel(feature_col, fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig)


def cluster_visualization(df: pd.DataFrame, cluster_col: str) -> None:
    """Main function to display various cluster visualizations.

    Args:
        df: DataFrame containing cluster data
        cluster_col: Name of the column containing cluster assignments
    """
    st.header("Cluster Visualizations")

    if df is None or df.empty:
        st.warning("Please upload cluster data to visualize")
        return

    if cluster_col not in df.columns:
        st.error(f"The uploaded data must contain a '{cluster_col}' column")
        return

    # Show cluster distribution
    st.subheader("Distribution Across Clusters")
    plot_cluster_distribution(df, cluster_col)

    # Feature analysis by cluster
    st.subheader("Feature Analysis by Cluster")

    # Get numeric columns for feature analysis
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != cluster_col]

    if numeric_cols:
        selected_feature = st.selectbox(
            "Select feature to analyze across clusters", options=numeric_cols
        )

        plot_feature_by_cluster(df, selected_feature, cluster_col)
    else:
        st.warning("No numeric features found for analysis")
