"""Feature exploration components for the clustering dashboard.

This module provides utilities for exploring and visualizing high-dimensional features.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def show_feature_distribution(
    data: pd.DataFrame,
    features: List[str],
    cluster_col: Optional[str] = None
) -> None:
    """Show distribution of selected features.
    
    Args:
        data: DataFrame with features
        features: List of feature columns
        cluster_col: Optional cluster column for coloring
    """
    # Let user select a feature to visualize
    selected_feature = st.selectbox(
        "Select feature to visualize",
        options=features
    )
    
    # Create histogram
    if cluster_col and cluster_col in data.columns:
        # Colored by cluster
        fig = px.histogram(
            data,
            x=selected_feature,
            color=cluster_col,
            marginal="box",
            title=f"Distribution of {selected_feature} by Cluster",
            labels={selected_feature: selected_feature, cluster_col: 'Cluster'}
        )
    else:
        # Single color
        fig = px.histogram(
            data,
            x=selected_feature,
            marginal="box",
            title=f"Distribution of {selected_feature}",
            labels={selected_feature: selected_feature}
        )
    
    st.plotly_chart(fig, use_container_width=True)


def run_pca(
    data: pd.DataFrame,
    features: List[str],
    cluster_col: str,
    n_components: int = 3
) -> Tuple[pd.DataFrame, List[float]]:
    """Run PCA on the selected features.
    
    Args:
        data: DataFrame with features
        features: List of feature columns
        cluster_col: Cluster column name
        n_components: Number of PCA components
        
    Returns:
        Tuple of (PCA results DataFrame, explained variance ratios)
    """
    # Create feature matrix
    X = data[features].values
    
    # Run PCA
    pca = PCA(n_components=min(n_components, len(features)))
    pca_result = pca.fit_transform(X)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        pca_result,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)]
    )
    
    # Add cluster information
    pca_df[cluster_col] = data[cluster_col].values
    
    # Add STORE_NBR if available
    if 'STORE_NBR' in data.columns:
        pca_df['STORE_NBR'] = data['STORE_NBR'].values
    
    return pca_df, pca.explained_variance_ratio_


def run_tsne(
    data: pd.DataFrame,
    features: List[str],
    cluster_col: str,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42
) -> pd.DataFrame:
    """Run t-SNE on the selected features.
    
    Args:
        data: DataFrame with features
        features: List of feature columns
        cluster_col: Cluster column name
        n_components: Number of t-SNE components
        perplexity: t-SNE perplexity parameter
        random_state: Random seed
        
    Returns:
        DataFrame with t-SNE results
    """
    # Create feature matrix
    X = data[features].values
    
    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(data) - 1),
        random_state=random_state
    )
    tsne_result = tsne.fit_transform(X)
    
    # Create DataFrame with t-SNE results
    tsne_df = pd.DataFrame(
        tsne_result,
        columns=[f'TSNE{i+1}' for i in range(n_components)]
    )
    
    # Add cluster information
    tsne_df[cluster_col] = data[cluster_col].values
    
    # Add STORE_NBR if available
    if 'STORE_NBR' in data.columns:
        tsne_df['STORE_NBR'] = data['STORE_NBR'].values
    
    return tsne_df


def show_dimensionality_reduction(
    data: pd.DataFrame,
    features: List[str],
    cluster_col: str
) -> None:
    """Show dimensionality reduction visualizations.
    
    Args:
        data: DataFrame with features
        features: List of feature columns
        cluster_col: Cluster column name
    """
    # Only proceed if we have enough features
    if len(features) < 3:
        st.warning("Need at least 3 features for dimensionality reduction")
        return
    
    # Let user select dimensionality reduction method
    method = st.selectbox(
        "Select dimensionality reduction method",
        options=["PCA", "t-SNE"],
        index=0
    )
    
    if method == "PCA":
        # Let user select number of components
        n_components = st.slider(
            "Number of PCA components",
            min_value=2,
            max_value=min(10, len(features)),
            value=3
        )
        
        # Run PCA
        with st.spinner("Running PCA..."):
            pca_df, explained_variance = run_pca(
                data,
                features,
                cluster_col,
                n_components
            )
        
        # Show explained variance
        st.write("### Explained Variance")
        
        # Create bar chart of explained variance
        fig = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_variance))],
            y=explained_variance,
            labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
            title="Explained Variance by Principal Component"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show PCA results
        if n_components >= 2:
            st.write("### PCA Visualization")
            
            # Show 2D or 3D plot
            if n_components >= 3:
                fig = px.scatter_3d(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    z="PC3",
                    color=cluster_col,
                    title="PCA Visualization (First 3 Components)",
                    labels={"PC1": "PC1", "PC2": "PC2", "PC3": "PC3", cluster_col: "Cluster"},
                    hover_data=['STORE_NBR'] if 'STORE_NBR' in pca_df.columns else None
                )
                fig.update_layout(height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    color=cluster_col,
                    title="PCA Visualization (First 2 Components)",
                    labels={"PC1": "PC1", "PC2": "PC2", cluster_col: "Cluster"},
                    hover_data=['STORE_NBR'] if 'STORE_NBR' in pca_df.columns else None
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif method == "t-SNE":
        # Let user select number of components
        n_components = st.slider(
            "Number of t-SNE components",
            min_value=2,
            max_value=3,
            value=2
        )
        
        # Let user adjust perplexity
        perplexity = st.slider(
            "Perplexity",
            min_value=5,
            max_value=min(50, len(data) - 1),
            value=min(30, len(data) // 3)
        )
        
        # Run t-SNE
        with st.spinner("Running t-SNE (this may take a while)..."):
            tsne_df = run_tsne(
                data,
                features,
                cluster_col,
                n_components,
                perplexity
            )
        
        # Show t-SNE results
        st.write("### t-SNE Visualization")
        
        # Show 2D or 3D plot
        if n_components == 3:
            fig = px.scatter_3d(
                tsne_df,
                x="TSNE1",
                y="TSNE2",
                z="TSNE3",
                color=cluster_col,
                title="t-SNE Visualization",
                labels={"TSNE1": "t-SNE 1", "TSNE2": "t-SNE 2", "TSNE3": "t-SNE 3", cluster_col: "Cluster"},
                hover_data=['STORE_NBR'] if 'STORE_NBR' in tsne_df.columns else None
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.scatter(
                tsne_df,
                x="TSNE1",
                y="TSNE2",
                color=cluster_col,
                title="t-SNE Visualization",
                labels={"TSNE1": "t-SNE 1", "TSNE2": "t-SNE 2", cluster_col: "Cluster"},
                hover_data=['STORE_NBR'] if 'STORE_NBR' in tsne_df.columns else None
            )
            st.plotly_chart(fig, use_container_width=True) 