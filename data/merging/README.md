# Merging Data Directory

This directory contains the output from the cluster merging process, which combines internal and external clustering results.

## Key Files

- `merged_cluster_assignments.pkl`: Final merged cluster assignments for all stores

## Process

The merging process:

1. Takes cluster assignments from both internal and external models
2. Creates a combined mapping of stores to their respective clusters
3. Handles any conflicts or special cases
4. Produces a unified cluster assignment
5. Saves the final result to `merged_cluster_assignments.pkl`

## Usage

The merged clusters provide a comprehensive view of store segmentation based on both internal sales patterns and external demographic/geographic factors. These final clusters are used for:

- Strategic decision making
- Marketing initiatives
- Merchandise planning
- Store network optimization

## Environment Variable

This directory is referenced by the environment variable `MERGING_DATA_DIR` with default value `/workspaces/testing-dagster/data/merging` 