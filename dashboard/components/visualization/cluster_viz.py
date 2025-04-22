"""Cluster visualization components."""

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


def plot_cluster_distribution(df: pd.DataFrame):
    """Plot the distribution of stores across clusters.

    Args:
        df: DataFrame containing cluster data with a 'CLUSTER' column
    """
    if "CLUSTER" not in df.columns:
        st.error("DataFrame must contain a 'CLUSTER' column")
        return

    # Count stores per cluster
    cluster_counts = df["CLUSTER"].value_counts().sort_index()

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

    ax.set_title("Number of Stores per Cluster", fontsize=16)
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Number of Stores", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    st.pyplot(fig)


def plot_feature_by_cluster(df: pd.DataFrame, feature_col: str):
    """Plot the distribution of a feature across clusters.

    Args:
        df: DataFrame containing cluster data
        feature_col: Column name of the feature to plot
    """
    if "CLUSTER" not in df.columns:
        st.error("DataFrame must contain a 'CLUSTER' column")
        return

    if feature_col not in df.columns:
        st.error(f"Feature column '{feature_col}' not found in DataFrame")
        return

    # Group by cluster and calculate statistics
    cluster_stats = df.groupby("CLUSTER")[feature_col].agg(["mean", "std"]).reset_index()

    # Create a bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    x = [f"Cluster {c}" for c in cluster_stats["CLUSTER"]]
    y = cluster_stats["mean"]
    err = cluster_stats["std"]

    bars = ax.bar(
        x, y, yerr=err, capsize=10, color=plt.cm.tab10.colors[: len(cluster_stats)], alpha=0.7
    )

    # Add mean value labels on top of bars
    for bar, mean_val in zip(bars, y):
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


def cluster_visualization(df: pd.DataFrame):
    """Main function to display various cluster visualizations.

    Args:
        df: DataFrame containing cluster data with a 'CLUSTER' column
    """
    st.header("Cluster Visualizations")

    if df is None or df.empty:
        st.warning("Please upload cluster data to visualize")
        return

    if "CLUSTER" not in df.columns:
        st.error("The uploaded data must contain a 'CLUSTER' column")
        return

    # Show cluster distribution
    st.subheader("Distribution of Stores Across Clusters")
    plot_cluster_distribution(df)

    # Feature analysis by cluster
    st.subheader("Feature Analysis by Cluster")

    # Get numeric columns for feature analysis
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "CLUSTER"]

    if numeric_cols:
        selected_feature = st.selectbox(
            "Select feature to analyze across clusters", options=numeric_cols
        )

        plot_feature_by_cluster(df, selected_feature)
    else:
        st.warning("No numeric features found for analysis")
