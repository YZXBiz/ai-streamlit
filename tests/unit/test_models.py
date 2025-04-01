"""Tests for the models module."""

import numpy as np

from clustering.core import models


def test_kmeans_clustering() -> None:
    """Test KMeans clustering functionality."""
    # Create sample data
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    # Create KMeans model with 2 clusters
    model = models.KMeansModel(n_clusters=2, random_state=42)

    # Fit the model
    model.fit(X)

    # Get the clusters
    clusters = model.predict(X)

    # Check we have 2 clusters
    assert len(np.unique(clusters)) == 2

    # Check the first 3 points are in one cluster and the last 3 in another
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] == clusters[4] == clusters[5]
    assert clusters[0] != clusters[3]


def test_hierarchical_clustering() -> None:
    """Test hierarchical clustering functionality."""
    # Create sample data
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    # Create hierarchical model with 2 clusters
    model = models.HierarchicalModel(n_clusters=2, linkage="ward")

    # Fit the model
    model.fit(X)

    # Get the clusters
    clusters = model.predict(X)

    # Check we have 2 clusters
    assert len(np.unique(clusters)) == 2

    # Check the first 3 points are in one cluster and the last 3 in another
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] == clusters[4] == clusters[5]
    assert clusters[0] != clusters[3]


def test_dbscan_clustering() -> None:
    """Test DBSCAN clustering functionality."""
    # Create sample data
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    # Create DBSCAN model
    model = models.DBSCANModel(eps=5, min_samples=2)

    # Fit the model
    model.fit(X)

    # Get the clusters
    clusters = model.predict(X)

    # Check we have 2 clusters
    assert len(np.unique(clusters)) == 2

    # Check the first 3 points are in one cluster and the last 3 in another
    assert clusters[0] == clusters[1] == clusters[2]
    assert clusters[3] == clusters[4] == clusters[5]
    assert clusters[0] != clusters[3]
