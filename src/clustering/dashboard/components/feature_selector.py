import streamlit as st
import pandas as pd
from typing import Any, Optional, Union
import re
import streamlit.components.v1 as components
import json
import html

def get_feature_type(df: pd.DataFrame, column: str) -> str:
    """
    Determine the type of a DataFrame column/feature.
    
    Args:
        df: The DataFrame containing the column
        column: The name of the column/feature
    
    Returns:
        A string describing the feature type ('numeric', 'categorical', 'datetime', etc.)
    """
    if column not in df.columns:
        return "unknown"
    
    dtype = df[column].dtype
    
    # Check for datetime features
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    
    # Check for numeric features
    elif pd.api.types.is_numeric_dtype(dtype):
        # Check if it's an integer column that's actually categorical
        if pd.api.types.is_integer_dtype(dtype) and df[column].nunique() < min(20, len(df) * 0.05):
            return "categorical"
        return "numeric"
    
    # Check for boolean features
    elif pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    
    # Everything else is considered categorical
    else:
        return "categorical"

def feature_selector(
    df: pd.DataFrame,
    title: str = "Feature Selection",
    default_x: Optional[str] = None,
    default_y: Optional[str] = None,
    default_color: Optional[str] = None,
    default_size: Optional[str] = None,
    allow_multiple: bool = False,
    key_prefix: str = ""
) -> dict[str, Any]:
    """
    Display an interactive feature selector with drag-and-drop functionality.
    
    Args:
        df: DataFrame containing the features to select from
        title: Title to display above the selector
        default_x: Default feature for X axis
        default_y: Default feature for Y axis
        default_color: Default feature for color encoding
        default_size: Default feature for size encoding
        allow_multiple: Whether to allow selection of multiple features
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
    
    Returns:
        Dictionary with selected features for various roles (x, y, color, etc.)
    """
    st.markdown(f"### {title}")
    
    # Get column types for filtering
    column_types = {col: get_feature_type(df, col) for col in df.columns}
    
    # Categorize columns
    numeric_cols = [col for col, type_ in column_types.items() if type_ == "numeric"]
    categorical_cols = [col for col, type_ in column_types.items() if type_ == "categorical" or type_ == "boolean"]
    datetime_cols = [col for col, type_ in column_types.items() if type_ == "datetime"]
    
    # Sort columns alphabetically within each category
    numeric_cols.sort()
    categorical_cols.sort()
    datetime_cols.sort()
    
    # Add column type indicators
    all_cols_with_types = []
    for col in numeric_cols:
        all_cols_with_types.append(f"{col} 🔢")
    for col in categorical_cols:
        all_cols_with_types.append(f"{col} 🔤")
    for col in datetime_cols:
        all_cols_with_types.append(f"{col} 📅")
    
    # Add search functionality
    search_term = st.text_input(
        "🔍 Search features", 
        key=f"{key_prefix}feature_search"
    )
    
    if search_term:
        filtered_cols = [c for c in all_cols_with_types if search_term.lower() in c.lower()]
    else:
        filtered_cols = all_cols_with_types
    
    # Create columns for the drag-and-drop interface
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Available Features")
        # Strip type indicators when returning selected features
        def strip_type_indicator(col_name):
            return re.sub(r' [🔢🔤📅]$', '', col_name)
        
        # Available features that can be dragged
        if allow_multiple:
            available_selected = st.multiselect(
                "Drag features from here",
                options=filtered_cols,
                default=[],
                key=f"{key_prefix}available_features"
            )
        else:
            available_selected = []
    
    with col2:
        st.markdown("#### Chart Configuration")
        
        # Get default values with type indicators
        default_x_with_type = None
        if default_x:
            for col in all_cols_with_types:
                if strip_type_indicator(col) == default_x:
                    default_x_with_type = col
                    break
        
        default_y_with_type = None
        if default_y:
            for col in all_cols_with_types:
                if strip_type_indicator(col) == default_y:
                    default_y_with_type = col
                    break
        
        default_color_with_type = None
        if default_color:
            for col in all_cols_with_types:
                if strip_type_indicator(col) == default_color:
                    default_color_with_type = col
                    break
        
        default_size_with_type = None
        if default_size:
            for col in all_cols_with_types:
                if strip_type_indicator(col) == default_size:
                    default_size_with_type = col
                    break
        
        # X-axis selector
        x_feature = st.selectbox(
            "X-axis feature",
            options=filtered_cols,
            index=filtered_cols.index(default_x_with_type) if default_x_with_type in filtered_cols else 0,
            key=f"{key_prefix}x_feature"
        )
        
        # Y-axis selector
        y_feature = st.selectbox(
            "Y-axis feature",
            options=filtered_cols,
            index=filtered_cols.index(default_y_with_type) if default_y_with_type in filtered_cols else min(1, len(filtered_cols)-1),
            key=f"{key_prefix}y_feature"
        )
        
        # Color encoding
        color_feature = st.selectbox(
            "Color by (optional)",
            options=["None"] + filtered_cols,
            index=filtered_cols.index(default_color_with_type) + 1 if default_color_with_type in filtered_cols else 0,
            key=f"{key_prefix}color_feature"
        )
        
        # Size encoding (for bubble charts, scatter plots)
        size_feature = st.selectbox(
            "Size by (optional)",
            options=["None"] + [c for c in filtered_cols if "🔢" in c],  # Only numeric features
            index=[c for c in filtered_cols if "🔢" in c].index(default_size_with_type) + 1 
                if default_size_with_type in [c for c in filtered_cols if "🔢" in c] else 0,
            key=f"{key_prefix}size_feature"
        )
    
    # Return clean feature names without type indicators
    result = {
        "x": strip_type_indicator(x_feature) if x_feature else None,
        "y": strip_type_indicator(y_feature) if y_feature else None,
        "color": strip_type_indicator(color_feature) if color_feature and color_feature != "None" else None,
        "size": strip_type_indicator(size_feature) if size_feature and size_feature != "None" else None
    }
    
    if allow_multiple:
        result["selected"] = [strip_type_indicator(f) for f in available_selected]
    
    return result

def chart_type_selector(key_prefix: str = "") -> dict[str, Any]:
    """
    Display a selector for chart type and options.
    
    Args:
        key_prefix: Prefix for Streamlit widget keys to avoid conflicts
    
    Returns:
        Dictionary with chart type and options
    """
    st.markdown("### Chart Configuration")
    
    # Chart types
    chart_types = [
        "Scatter Plot",
        "3D Scatter",
        "Bubble Chart",
        "Line Plot",
        "Bar Chart",
        "Box Plot",
        "Violin Plot",
        "Histogram",
        "Density Heatmap"
    ]
    
    chart_type = st.selectbox(
        "Chart Type",
        options=chart_types,
        index=0,
        key=f"{key_prefix}chart_type"
    )
    
    # Options based on chart type
    options = {}
    
    if chart_type in ["Scatter Plot", "Bubble Chart", "3D Scatter"]:
        options["trendline"] = st.checkbox("Show trendline", value=False, key=f"{key_prefix}trendline")
        options["markers"] = st.checkbox("Show markers", value=True, key=f"{key_prefix}markers")
    
    if chart_type in ["Scatter Plot", "Bubble Chart", "3D Scatter", "Line Plot"]:
        options["log_x"] = st.checkbox("Log scale (X-axis)", value=False, key=f"{key_prefix}log_x")
        options["log_y"] = st.checkbox("Log scale (Y-axis)", value=False, key=f"{key_prefix}log_y")
    
    if chart_type == "3D Scatter":
        # Z-axis is only relevant for 3D Scatter
        options["show_z"] = True
    else:
        options["show_z"] = False
    
    if chart_type in ["Histogram", "Density Heatmap"]:
        options["normalize"] = st.checkbox("Normalize", value=False, key=f"{key_prefix}normalize")
        options["cumulative"] = st.checkbox("Cumulative", value=False, key=f"{key_prefix}cumulative")
    
    if chart_type == "Histogram":
        options["bins"] = st.slider("Number of bins", min_value=5, max_value=100, value=20, key=f"{key_prefix}bins")
    
    return {
        "chart_type": chart_type,
        "options": options
    }

def display_column_info(df: pd.DataFrame, column: str) -> None:
    """
    Display information about a dataframe column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to display information for
    """
    col_type = df[column].dtype
    
    # Get basic statistics based on column type
    if pd.api.types.is_numeric_dtype(df[column]):
        stats = {
            "Type": str(col_type),
            "Min": f"{df[column].min():.4g}",
            "Max": f"{df[column].max():.4g}",
            "Mean": f"{df[column].mean():.4g}",
            "Median": f"{df[column].median():.4g}",
            "Std Dev": f"{df[column].std():.4g}",
            "Unique Values": df[column].nunique(),
            "Missing Values": df[column].isna().sum(),
        }
    else:
        # For categorical, datetime, or other types
        stats = {
            "Type": str(col_type),
            "Unique Values": df[column].nunique(),
            "Missing Values": df[column].isna().sum(),
            "Most Common": f"{df[column].value_counts().index[0] if not df[column].value_counts().empty else 'None'}",
            "Most Common Count": f"{df[column].value_counts().iloc[0] if not df[column].value_counts().empty else 0}",
        }
    
    # Display stats in a nice format
    st.write("#### Column Statistics")
    col1, col2 = st.columns(2)
    
    for i, (k, v) in enumerate(stats.items()):
        if i % 2 == 0:
            col1.metric(k, v)
        else:
            col2.metric(k, v)
    
    # Show a sample of values
    st.write("#### Sample Values")
    sample_values = df[column].sample(min(5, len(df))).tolist()
    st.text(", ".join(str(val) for val in sample_values))
    
    # If numeric, show a mini histogram
    if pd.api.types.is_numeric_dtype(df[column]):
        st.write("#### Distribution")
        st.bar_chart(df[column].value_counts())

def create_drag_drop_interface(columns: list[str], 
                               selected_features: list[str]) -> dict[str, list[str]]:
    """
    Creates a drag-and-drop interface for selecting features.
    
    Args:
        columns: List of all available columns
        selected_features: Currently selected features
        
    Returns:
        Dictionary with available and selected features
    """
    # Prepare available columns (excluding those already selected)
    available_columns = [col for col in columns if col not in selected_features]
    
    # Create custom HTML/JS for drag and drop
    html_code = f"""
    <style>
        .column-container {{
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            min-height: 50px;
            background-color: #f9f9f9;
        }}
        .column-item {{
            padding: 8px;
            margin: 5px;
            background-color: #e0f7fa;
            border: 1px solid #4fc3f7;
            border-radius: 4px;
            cursor: grab;
            display: inline-block;
        }}
        .selected-container {{
            background-color: #e8f5e9;
            border: 1px solid #81c784;
        }}
        .container-title {{
            font-weight: bold;
            margin-bottom: 10px;
        }}
        #selected-columns-container .column-item {{
            background-color: #c8e6c9;
            border: 1px solid #66bb6a;
        }}
    </style>
    
    <div>
        <div class="container-title">Available Columns (drag to select)</div>
        <div id="available-columns-container" class="column-container">
            {" ".join([f'<div class="column-item" draggable="true" data-column="{html.escape(col)}">{html.escape(col)}</div>' for col in available_columns])}
        </div>
        
        <div class="container-title">Selected Features (drag to reorder or remove)</div>
        <div id="selected-columns-container" class="column-container selected-container">
            {" ".join([f'<div class="column-item" draggable="true" data-column="{html.escape(col)}">{html.escape(col)}</div>' for col in selected_features])}
        </div>
    </div>
    
    <script>
        // Store the current state
        let availableColumns = {json.dumps(available_columns)};
        let selectedColumns = {json.dumps(selected_features)};
        
        // Track which element is being dragged
        let draggedItem = null;
        let sourceContainer = null;
        
        // Set up event listeners for all draggable items
        document.querySelectorAll('.column-item').forEach(item => {{
            item.addEventListener('dragstart', handleDragStart);
            item.addEventListener('dragend', handleDragEnd);
        }});
        
        // Set up container event listeners
        const containers = document.querySelectorAll('.column-container');
        containers.forEach(container => {{
            container.addEventListener('dragover', handleDragOver);
            container.addEventListener('dragenter', handleDragEnter);
            container.addEventListener('dragleave', handleDragLeave);
            container.addEventListener('drop', handleDrop);
        }});
        
        function handleDragStart(e) {{
            draggedItem = e.target;
            sourceContainer = e.target.parentNode;
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', e.target.dataset.column);
            e.target.style.opacity = '0.4';
        }}
        
        function handleDragEnd(e) {{
            e.target.style.opacity = '1';
            
            // Update the internal state
            availableColumns = Array.from(
                document.querySelector('#available-columns-container').querySelectorAll('.column-item')
            ).map(item => item.dataset.column);
            
            selectedColumns = Array.from(
                document.querySelector('#selected-columns-container').querySelectorAll('.column-item')
            ).map(item => item.dataset.column);
            
            // Send the updated data to Streamlit
            sendDataToStreamlit();
        }}
        
        function handleDragOver(e) {{
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
        }}
        
        function handleDragEnter(e) {{
            e.target.classList.add('drag-over');
        }}
        
        function handleDragLeave(e) {{
            e.target.classList.remove('drag-over');
        }}
        
        function handleDrop(e) {{
            e.preventDefault();
            const container = e.target.closest('.column-container');
            if (!container) return;
            
            container.classList.remove('drag-over');
            
            // Only proceed if we're dropping onto a different container
            if (sourceContainer !== container) {{
                // Remove the item from its original container
                sourceContainer.removeChild(draggedItem);
                
                // Add it to the new container
                container.appendChild(draggedItem);
            }}
        }}
        
        function sendDataToStreamlit() {{
            const data = {{
                available: availableColumns,
                selected: selectedColumns
            }};
            
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: data
            }}, '*');
        }}
    </script>
    """
    
    # Use a unique key for the component to prevent caching issues
    key = f"feature_selector_{','.join(selected_features)}"
    
    # Render the component and get the selected features
    component_value = components.html(html_code, height=300, key=key)
    
    # Default return value if the component hasn't sent a message yet
    if component_value is None:
        return {"available": available_columns, "selected": selected_features}
    
    return component_value

def select_features(df: pd.DataFrame) -> list[str]:
    """
    Allows users to select features from a dataframe using a drag-and-drop interface.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        List of selected column names
    """
    if df is None or df.empty:
        st.warning("Please load a dataset first")
        return []
    
    st.write("## Feature Selection")
    st.write("Drag columns to select features for visualization")
    
    # Get columns grouped by data type
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]
    
    # Initialize session state for selected features if it doesn't exist
    if 'selected_features' not in st.session_state:
        # Start with some sensible defaults (e.g., first two numeric columns)
        default_selected = numeric_cols[:min(2, len(numeric_cols))]
        st.session_state.selected_features = default_selected
    
    # Tabs for column type filtering
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "All Columns", 
        f"Numeric ({len(numeric_cols)})", 
        f"Categorical ({len(categorical_cols)})",
        f"Datetime ({len(datetime_cols)})",
        f"Other ({len(other_cols)})"
    ])
    
    with tab1:
        result = create_drag_drop_interface(df.columns.tolist(), st.session_state.selected_features)
        st.session_state.selected_features = result.get("selected", st.session_state.selected_features)
    
    with tab2:
        if numeric_cols:
            result = create_drag_drop_interface(numeric_cols, 
                                               [col for col in st.session_state.selected_features if col in numeric_cols])
        else:
            st.info("No numeric columns in this dataset")
            
    with tab3:
        if categorical_cols:
            result = create_drag_drop_interface(categorical_cols,
                                               [col for col in st.session_state.selected_features if col in categorical_cols])
        else:
            st.info("No categorical columns in this dataset")
            
    with tab4:
        if datetime_cols:
            result = create_drag_drop_interface(datetime_cols,
                                               [col for col in st.session_state.selected_features if col in datetime_cols])
        else:
            st.info("No datetime columns in this dataset")
            
    with tab5:
        if other_cols:
            result = create_drag_drop_interface(other_cols,
                                               [col for col in st.session_state.selected_features if col in other_cols])
        else:
            st.info("No other columns in this dataset")
    
    # Column info section
    st.write("## Column Details")
    if st.session_state.selected_features:
        selected_column = st.selectbox(
            "Select a column to view details",
            options=st.session_state.selected_features
        )
        
        if selected_column:
            display_column_info(df, selected_column)
    
    return st.session_state.selected_features

def feature_stats(df: pd.DataFrame, features: list[str]) -> None:
    """
    Display statistics for selected features.
    
    Args:
        df: DataFrame containing the data
        features: List of selected feature names
    """
    if not features or df is None:
        return
    
    st.write("### Feature Statistics")
    
    for feature in features:
        if feature not in df.columns:
            continue
            
        st.write(f"**{feature}**")
        
        if pd.api.types.is_numeric_dtype(df[feature]):
            stats = df[feature].describe()
            st.write(f"Min: {stats['min']:.2f}, Max: {stats['max']:.2f}, Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
            
            # Show distribution
            try:
                st.line_chart(df[feature].value_counts().sort_index())
            except:
                st.write("Could not plot distribution.")
        else:
            # For categorical data
            value_counts = df[feature].value_counts()
            st.write(f"Unique values: {len(value_counts)}")
            
            # Show top categories
            if len(value_counts) <= 10:
                st.bar_chart(value_counts)
            else:
                st.bar_chart(value_counts.head(10))
                st.write(f"(Showing top 10 of {len(value_counts)} categories)")

def get_all_selected_features(selected_features: dict[str, list[str]]) -> list[str]:
    """
    Get a flat list of all selected features.
    
    Args:
        selected_features: Dictionary with selected features for different chart dimensions
        
    Returns:
        List of unique selected feature names
    """
    all_features = []
    for features in selected_features.values():
        all_features.extend(features)
    
    return list(set(all_features))  # Remove duplicates 