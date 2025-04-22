"""
Smart Visualization Component for the Dashboard.

This module provides a component that intelligently suggests and renders visualizations
based on query results and data patterns.
"""
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure

from app.models.agent import VisualizationSuggester
from app.models.query_planner import QueryPlan, QueryType


def smart_viz(
    results: List[Dict[str, Any]],
    query: str = "",
    query_plan: Optional[QueryPlan] = None,
    schema: Optional[Dict[str, Any]] = None
) -> None:
    """
    Smart visualization component that suggests and renders the most appropriate visualization.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        Query results to visualize
    query : str, default=""
        The original natural language query
    query_plan : Optional[QueryPlan], default=None
        The query plan used to generate the SQL
    schema : Optional[Dict[str, Any]], default=None
        Schema information for the data
    
    Returns
    -------
    None
        This function displays visualizations in the Streamlit UI
    
    Notes
    -----
    This component:
    1. Uses AI to suggest the most appropriate visualization type
    2. Configures visualization parameters automatically
    3. Renders the visualization with sensible defaults
    4. Allows users to refine the visualization
    """
    if not results:
        st.info("No results to visualize.")
        return
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Create default query plan if none provided
    if query_plan is None:
        query_plan = QueryPlan(query_type=QueryType.FILTER)
    
    # Use default schema if none provided
    if schema is None:
        schema = {"columns": {col: {"type": str(dtype)} for col, dtype in df.dtypes.items()}}
    
    # Get visualization suggestions
    viz_config = VisualizationSuggester.suggest_visualization(
        query=query,
        plan=query_plan,
        results=results,
        schema=schema
    )
    
    # Display visualization title
    if viz_config["title"]:
        st.subheader(viz_config["title"])
    
    # Show description to the user
    st.caption(viz_config["description"])
    
    # Render the visualization based on type
    with st.container():
        fig = _create_visualization(df, viz_config)
        
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback to table view if visualization creation failed
            st.dataframe(df, use_container_width=True)
    
    # Show recommendations if available
    if "recommendations" in viz_config and viz_config["recommendations"]:
        st.write("Recommendations:")
        for rec in viz_config["recommendations"]:
            st.write(f"• Try a {rec['type']} visualization: {rec['reason']}")
    
    # Advanced options (collapsed by default)
    with st.expander("Customize Visualization", expanded=False):
        _customize_visualization(df, viz_config)


def _create_visualization(df: pd.DataFrame, config: Dict[str, Any]) -> Optional[Figure]:
    """
    Create a visualization based on the configuration.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data to visualize
    config : Dict[str, Any]
        Visualization configuration
    
    Returns
    -------
    Optional[Figure]
        Plotly figure object or None if visualization creation failed
    """
    viz_type = config["type"]
    
    try:
        if viz_type == "table" or viz_type is None:
            # Table view is handled directly in the parent function
            return None
        
        elif viz_type == "bar":
            # Handle missing x_axis or y_axis with sensible defaults
            x_axis = config["x_axis"]
            y_axis = config["y_axis"]
            
            if x_axis is None:
                # Find a categorical column for x_axis
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns
                if not categorical_cols.empty:
                    x_axis = categorical_cols[0]
                else:
                    # Fallback to first column
                    x_axis = df.columns[0]
            
            if y_axis is None:
                # Find a numeric column for y_axis
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    y_axis = numeric_cols[0]
                else:
                    # Use count aggregation if no numeric column
                    y_axis = "count"
                    df = df[x_axis].value_counts().reset_index()
                    df.columns = [x_axis, y_axis]
            
            orientation = config.get("orientation", "vertical")
            
            if orientation == "horizontal":
                fig = px.bar(
                    df, y=x_axis, x=y_axis, 
                    title=config.get("title", ""),
                    color=config.get("color_by")
                )
            else:
                fig = px.bar(
                    df, x=x_axis, y=y_axis, 
                    title=config.get("title", ""),
                    color=config.get("color_by")
                )
            
            # Apply sorting if specified
            if config.get("sort"):
                if config["sort"] == "ASC":
                    fig.update_layout(xaxis={'categoryorder': 'total ascending'})
                else:
                    fig.update_layout(xaxis={'categoryorder': 'total descending'})
            
            return fig
        
        elif viz_type == "line":
            x_axis = config["x_axis"] or df.columns[0]
            y_axis = config["y_axis"]
            
            if y_axis is None:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    y_axis = numeric_cols[0]
                else:
                    return None
                    
            return px.line(
                df, x=x_axis, y=y_axis, 
                title=config.get("title", ""),
                color=config.get("color_by")
            )
        
        elif viz_type == "scatter":
            # Scatter plot needs two numeric columns
            x_axis = config["x_axis"]
            y_axis = config["y_axis"]
            
            if x_axis is None or y_axis is None:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if len(numeric_cols) >= 2:
                    x_axis = numeric_cols[0]
                    y_axis = numeric_cols[1]
                else:
                    return None
            
            return px.scatter(
                df, x=x_axis, y=y_axis, 
                title=config.get("title", ""),
                color=config.get("color_by"),
                size=config.get("size_by")
            )
        
        elif viz_type == "pie":
            x_axis = config["x_axis"]
            y_axis = config["y_axis"]
            
            if x_axis is None:
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns
                if not categorical_cols.empty:
                    x_axis = categorical_cols[0]
                else:
                    return None
            
            if y_axis is None:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    y_axis = numeric_cols[0]
                else:
                    # Use counts if no values column
                    counts = df[x_axis].value_counts().reset_index()
                    counts.columns = [x_axis, 'count']
                    return px.pie(counts, names=x_axis, values='count', title=config.get("title", ""))
            
            return px.pie(df, names=x_axis, values=y_axis, title=config.get("title", ""))
        
        elif viz_type == "histogram":
            x_axis = config["x_axis"]
            
            if x_axis is None:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    x_axis = numeric_cols[0]
                else:
                    return None
            
            return px.histogram(
                df, x=x_axis, 
                title=config.get("title", ""),
                color=config.get("color_by")
            )
        
        elif viz_type == "box":
            y_axis = config["y_axis"]
            
            if y_axis is None:
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    y_axis = numeric_cols[0]
                else:
                    return None
            
            return px.box(
                df, y=y_axis, x=config.get("x_axis"),
                title=config.get("title", "")
            )
        
        elif viz_type == "heatmap":
            # For heatmap, we need a pivot table
            x_axis = config["x_axis"]
            y_axis = config["y_axis"]
            z_axis = config.get("color_by")
            
            if not all([x_axis, y_axis, z_axis]):
                return None
            
            pivot_df = df.pivot_table(
                values=z_axis, 
                index=y_axis, 
                columns=x_axis, 
                aggfunc=config.get("aggregation", "mean")
            )
            
            return px.imshow(
                pivot_df,
                title=config.get("title", "")
            )
        
        elif viz_type == "map":
            # Map visualization requires latitude and longitude
            lat = config["y_axis"]  # Usually latitude
            lon = config["x_axis"]  # Usually longitude
            
            if lat is None or lon is None:
                # Look for columns with lat/lon in their names
                lat_cols = [col for col in df.columns if "lat" in col.lower()]
                lon_cols = [col for col in df.columns if "lon" in col.lower()]
                
                if lat_cols and lon_cols:
                    lat = lat_cols[0]
                    lon = lon_cols[0]
                else:
                    return None
            
            return px.scatter_mapbox(
                df, lat=lat, lon=lon,
                color=config.get("color_by"),
                size=config.get("size_by"),
                title=config.get("title", ""),
                mapbox_style="open-street-map"
            )
        
        elif viz_type == "kpi":
            # KPI is a single value
            value_col = config["y_axis"]
            
            if value_col is None:
                # Find first numeric column
                numeric_cols = df.select_dtypes(include=["number"]).columns
                if not numeric_cols.empty:
                    value_col = numeric_cols[0]
                else:
                    return None
            
            # Get the value (assuming first row if multiple)
            value = df[value_col].iloc[0] if len(df) > 0 else 0
            
            # Create a simple gauge or indicator
            fig = go.Figure(go.Indicator(
                mode="number",
                value=value,
                title={"text": config.get("title", value_col)}
            ))
            
            fig.update_layout(height=250)
            return fig
        
        # Add more visualization types as needed
            
        return None
    
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None


def _customize_visualization(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Provide UI controls to customize the visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data
    config : Dict[str, Any]
        Current visualization configuration
    
    Returns
    -------
    None
        Updates are reflected in the Streamlit UI
    """
    st.write("Coming soon: Advanced visualization customization options")
    
    # Example of future customization options:
    # - Change chart type
    # - Modify axes and labels
    # - Adjust colors and themes
    # - Set aggregation methods
    # - Add trend lines or annotations 