"""Feature explorer components for the clustering dashboard.

This module provides components for exploring feature distributions and relationships.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Optional, Union, Any

from clustering.dashboard.utils import get_color_scale


def show_feature_distribution(
    data: pd.DataFrame, 
    features: list[str], 
    cluster_col: Optional[str] = None
) -> None:
    """Show distribution of selected features.
    
    Args:
        data: DataFrame with features and cluster column
        features: List of feature columns to display
        cluster_col: Name of the cluster column (optional)
    """
    st.markdown("### Feature Distribution")
    
    # Let user select a feature
    feature = st.selectbox("Select Feature", features)
    
    # Let user select chart type
    chart_type = st.selectbox(
        "Chart Type",
        ["Histogram", "Box Plot", "Violin Plot", "KDE Plot"],
        index=0
    )
    
    # Create plot based on chart type
    if chart_type == "Histogram":
        fig = px.histogram(
            data, 
            x=feature,
            color=cluster_col,
            marginal="box",
            title=f"Distribution of {feature}",
            opacity=0.7
        )
    
    elif chart_type == "Box Plot":
        fig = px.box(
            data,
            y=feature,
            x=cluster_col if cluster_col else None,
            title=f"Box Plot of {feature}",
            color=cluster_col
        )
    
    elif chart_type == "Violin Plot":
        fig = px.violin(
            data,
            y=feature,
            x=cluster_col if cluster_col else None,
            title=f"Violin Plot of {feature}",
            color=cluster_col,
            box=True
        )
    
    elif chart_type == "KDE Plot":
        fig = px.density_contour(
            data,
            x=feature,
            title=f"KDE Plot of {feature}",
            color=cluster_col
        )
        fig.update_traces(contours_coloring="fill", contours_showlabels=True)
    
    # Update layout
    fig.update_layout(
        xaxis_title=feature,
        yaxis_title="Count" if chart_type == "Histogram" else feature,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FFFFFF"),
        title_font=dict(size=20, color="#FFFFFF"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show feature statistics
    with st.expander("Feature Statistics"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Basic Statistics")
            st.write(data[feature].describe())
        
        with col2:
            if cluster_col:
                st.write("#### Statistics by Cluster")
                st.write(data.groupby(cluster_col)[feature].describe())


def run_pca(
    data: pd.DataFrame, 
    features: list[str], 
    n_components: int = 2
) -> tuple[pd.DataFrame, list[float]]:
    """Run PCA on the selected features.
    
    Args:
        data: DataFrame with features
        features: List of feature columns for PCA
        n_components: Number of principal components
        
    Returns:
        Tuple containing the transformed DataFrame and explained variance ratios
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Extract and standardize features
    X = data[features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a DataFrame with PCA results
    pca_df = pd.DataFrame(
        X_pca, 
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    
    # Add index from original data
    pca_df.index = data.index
    
    return pca_df, pca.explained_variance_ratio_.tolist()


def run_tsne(
    data: pd.DataFrame, 
    features: list[str], 
    n_components: int = 2, 
    perplexity: int = 30
) -> pd.DataFrame:
    """Run t-SNE on the selected features.
    
    Args:
        data: DataFrame with features
        features: List of feature columns for t-SNE
        n_components: Number of dimensions for t-SNE
        perplexity: t-SNE perplexity parameter
        
    Returns:
        DataFrame with t-SNE results
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    
    # Extract and standardize features
    X = data[features].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=300,
        random_state=42
    )
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Create a DataFrame with t-SNE results
    tsne_df = pd.DataFrame(
        X_tsne, 
        columns=[f"t-SNE{i+1}" for i in range(n_components)]
    )
    
    # Add index from original data
    tsne_df.index = data.index
    
    return tsne_df


def show_dimensionality_reduction(
    data: pd.DataFrame, 
    features: list[str], 
    cluster_col: Optional[str] = None
) -> None:
    """Show dimensionality reduction visualization.
    
    Args:
        data: DataFrame with features and clusters
        features: List of feature columns
        cluster_col: Name of the cluster column (optional)
    """
    if len(features) < 3:
        st.warning("Need at least 3 features for dimensionality reduction")
        return
        
    st.markdown("### Dimensionality Reduction")
    
    # Let user select technique
    technique = st.selectbox(
        "Technique", 
        ["PCA", "t-SNE"],
        index=0
    )
    
    # Let user select number of components
    n_components = st.slider(
        "Number of Components", 
        min_value=2, 
        max_value=min(5, len(features)),
        value=2
    )
    
    # Additional parameters for t-SNE
    perplexity = 30
    if technique == "t-SNE":
        perplexity = st.slider(
            "Perplexity", 
            min_value=5, 
            max_value=50,
            value=30
        )
    
    # Run dimensionality reduction
    if technique == "PCA":
        with st.spinner("Running PCA..."):
            result_df, explained_var = run_pca(
                data, 
                features, 
                n_components=n_components
            )
            
            # Display explained variance
            st.write("#### Explained Variance")
            explained_df = pd.DataFrame({
                'Component': [f"PC{i+1}" for i in range(n_components)],
                'Explained Variance Ratio': explained_var,
                'Cumulative Explained Variance': np.cumsum(explained_var)
            })
            st.dataframe(explained_df)
            
    else:  # t-SNE
        with st.spinner("Running t-SNE (this may take a while)..."):
            result_df = run_tsne(
                data, 
                features, 
                n_components=n_components,
                perplexity=perplexity
            )
    
    # Visualization
    if n_components == 2:
        # 2D visualization
        fig = px.scatter(
            result_df,
            x=result_df.columns[0],
            y=result_df.columns[1],
            color=data[cluster_col] if cluster_col else None,
            title=f"{technique} Visualization",
            labels={
                result_df.columns[0]: result_df.columns[0],
                result_df.columns[1]: result_df.columns[1]
            }
        )
        
    elif n_components == 3:
        # 3D visualization
        fig = px.scatter_3d(
            result_df,
            x=result_df.columns[0],
            y=result_df.columns[1],
            z=result_df.columns[2],
            color=data[cluster_col] if cluster_col else None,
            title=f"{technique} Visualization",
            labels={
                result_df.columns[0]: result_df.columns[0],
                result_df.columns[1]: result_df.columns[1],
                result_df.columns[2]: result_df.columns[2]
            }
        )
        
    else:
        # Parallel coordinates for more than 3 dimensions
        # Combine result with cluster information
        combined = result_df.copy()
        if cluster_col:
            combined[cluster_col] = data[cluster_col].values
            
        fig = px.parallel_coordinates(
            combined,
            color=cluster_col if cluster_col else None,
            dimensions=result_df.columns.tolist(),
            title=f"{technique} Visualization"
        )
    
    # Update layout for better appearance
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=50, b=20),
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the top contributing features for PCA
    if technique == "PCA" and st.checkbox("Show feature contributions"):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # Extract and scale features
        X = data[features].copy().fillna(data[features].mean())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        # Show loading scores for each component
        for i in range(n_components):
            st.write(f"#### Principal Component {i+1}")
            
            # Get loading scores
            loading_scores = pd.DataFrame(
                pca.components_[i], 
                columns=["Loading Score"],
                index=features
            )
            loading_scores["Abs Loading Score"] = loading_scores["Loading Score"].abs()
            loading_scores = loading_scores.sort_values("Abs Loading Score", ascending=False)
            
            # Plot top features
            top_features = loading_scores.head(10)
            
            fig = px.bar(
                top_features,
                y=top_features.index,
                x="Loading Score",
                orientation='h',
                title=f"Top Features Contributing to PC{i+1}"
            )
            
            fig.update_layout(
                yaxis_title="Feature",
                xaxis_title="Loading Score",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True) 