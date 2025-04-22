"""
Pydantic models and agent implementation for the chat interface.

This module defines the agent structure and workflow for converting natural language
queries to SQL and providing results with explanations.
"""
import re
import json
import time
from enum import Enum
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from pydantic import BaseModel, Field, validator

# Import the logging utilities
from app.utils import get_logger, log_operation, mdc_context

# Set up logger
logger = get_logger(__name__)

class QueryType(str, Enum):
    """Types of queries that can be handled."""
    SELECT = "select"
    AGGREGATE = "aggregate"
    FILTER = "filter" 
    GROUP_BY = "group_by"
    ORDER_BY = "order_by"
    LIMIT = "limit"
    JOIN = "join"
    COMPLEX = "complex"


class UserQuery(BaseModel):
    """Model for user query with context."""
    query: str = Field(..., description="User's natural language query about the data")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context including data schema")


class QueryPlan(BaseModel):
    """Represents the structured plan for a query before SQL generation."""
    query_type: QueryType = Field(..., description="Type of the query")
    select_columns: List[str] = Field(default_factory=list, description="Columns to select")
    where_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Filter conditions")
    group_by_columns: List[str] = Field(default_factory=list, description="Columns to group by")
    order_by: List[Dict[str, str]] = Field(default_factory=list, description="Order by specifications")
    limit: Optional[int] = Field(None, description="Limit on number of rows")
    aggregations: List[Dict[str, Any]] = Field(default_factory=list, description="Aggregation operations")
    table_name: str = Field(..., description="Table to query")
    joins: List[Dict[str, Any]] = Field(default_factory=list, description="Join specifications")
    
    @validator('where_conditions')
    def validate_conditions(cls, v):
        """Validate that conditions have the required fields."""
        for condition in v:
            if not all(k in condition for k in ['column', 'operator', 'value']):
                raise ValueError("Conditions must contain column, operator, and value")
        return v


class QueryResponse(BaseModel):
    """Model for query response."""
    sql: str = Field(..., description="Generated SQL query")
    result: List[Dict[str, Any]] = Field(..., description="Query results as list of dictionaries")
    explanation: str = Field(..., description="Natural language explanation of results")
    visualization_type: Optional[str] = Field(None, description="Suggested visualization type")
    query_time_ms: int = Field(0, description="Query execution time in milliseconds")
    row_count: int = Field(0, description="Number of rows in the result")
    column_insights: Dict[str, Any] = Field(default_factory=dict, description="Insights about specific columns")


class SQLGenerator:
    """Handles generation of SQL from structured query plans."""
    
    @staticmethod
    def generate_sql(plan: QueryPlan) -> str:
        """
        Generate SQL from a query plan.
        
        Args:
            plan: The structured query plan
            
        Returns:
            SQL query string
        """
        with log_operation(logger, "sql_generation"):
            # Start building the SQL query
            sql_parts = ["SELECT"]
            
            # Handle SELECT clause
            if plan.aggregations:
                select_expressions = []
                
                for agg in plan.aggregations:
                    func = agg.get('function', '').upper()
                    col = agg.get('column', '*')
                    alias = agg.get('alias', f"{func.lower()}_{col}")
                    
                    # Handle COUNT(*) separately
                    if func == 'COUNT' and col == '*':
                        select_expressions.append(f"{func}(*) AS {alias}")
                    else:
                        select_expressions.append(f"{func}({col}) AS {alias}")
                
                # Add any non-aggregated columns that are in GROUP BY
                for col in plan.group_by_columns:
                    if col not in [a.get('column') for a in plan.aggregations]:
                        select_expressions.append(col)
                
                sql_parts.append(", ".join(select_expressions))
            elif plan.select_columns:
                sql_parts.append(", ".join(plan.select_columns))
            else:
                sql_parts.append("*")
            
            # FROM clause
            sql_parts.append(f"FROM {plan.table_name}")
            
            # JOIN clause
            for join in plan.joins:
                join_type = join.get('type', 'INNER').upper()
                table = join.get('table')
                condition = join.get('condition')
                if table and condition:
                    sql_parts.append(f"{join_type} JOIN {table} ON {condition}")
            
            # WHERE clause
            if plan.where_conditions:
                where_clauses = []
                for condition in plan.where_conditions:
                    column = condition.get('column')
                    operator = condition.get('operator', '=')
                    value = condition.get('value')
                    
                    # Format value based on type
                    if isinstance(value, str):
                        formatted_value = f"'{value}'"
                    else:
                        formatted_value = str(value)
                    
                    where_clauses.append(f"{column} {operator} {formatted_value}")
                
                sql_parts.append("WHERE " + " AND ".join(where_clauses))
            
            # GROUP BY clause
            if plan.group_by_columns:
                sql_parts.append("GROUP BY " + ", ".join(plan.group_by_columns))
            
            # ORDER BY clause
            if plan.order_by:
                order_parts = []
                for order_spec in plan.order_by:
                    col = order_spec.get('column')
                    direction = order_spec.get('direction', 'ASC').upper()
                    order_parts.append(f"{col} {direction}")
                
                sql_parts.append("ORDER BY " + ", ".join(order_parts))
            
            # LIMIT clause
            if plan.limit is not None:
                sql_parts.append(f"LIMIT {plan.limit}")
            
            # Join all parts with spaces
            sql = " ".join(sql_parts)
            
            logger.info(f"Generated SQL: {sql}")
            return sql


class NLQueryAnalyzer:
    """Analyzes natural language queries to extract query components."""
    
    # Common patterns for different query types
    AGGREGATION_KEYWORDS = {
        'count': 'COUNT',
        'how many': 'COUNT',
        'total number': 'COUNT',
        'average': 'AVG',
        'avg': 'AVG',
        'mean': 'AVG',
        'sum': 'SUM',
        'total sum': 'SUM',
        'total amount': 'SUM',
        'maximum': 'MAX',
        'max': 'MAX',
        'highest': 'MAX',
        'largest': 'MAX',
        'minimum': 'MIN',
        'min': 'MIN',
        'lowest': 'MIN',
        'smallest': 'MIN'
    }
    
    ORDER_KEYWORDS = {
        'ascending': 'ASC',
        'increasing': 'ASC',
        'descending': 'DESC',
        'decreasing': 'DESC',
        'highest to lowest': 'DESC',
        'largest to smallest': 'DESC',
        'biggest to smallest': 'DESC',
        'most to least': 'DESC',
        'greatest to least': 'DESC',
        'lowest to highest': 'ASC',
        'smallest to largest': 'ASC',
        'least to most': 'ASC',
    }
    
    FILTER_KEYWORDS = {
        'where': '=',
        'with': '=',
        'greater than': '>',
        'more than': '>',
        'higher than': '>',
        'above': '>',
        'greater than or equal to': '>=',
        'at least': '>=',
        'less than': '<',
        'lower than': '<',
        'below': '<',
        'less than or equal to': '<=',
        'at most': '<=',
        'equal to': '=',
        'equals': '=',
        'is': '=',
        'not equal to': '!=',
        'not': '!=',
        'between': 'BETWEEN',
        'in': 'IN',
        'like': 'LIKE',
        'contains': 'LIKE',
        'starts with': 'LIKE',
        'ends with': 'LIKE'
    }
    
    @classmethod
    def analyze_query(cls, query: str, schema: Dict[str, Any]) -> QueryPlan:
        """
        Analyze a natural language query and extract components.
        
        Args:
            query: Natural language query
            schema: Schema information for the data
            
        Returns:
            Structured query plan
        """
        with log_operation(logger, "nl_query_analysis"):
            query_lower = query.lower()
            table_name = schema.get('table_name', 'data')
            columns = list(schema.get('columns', {}).keys())
            
            # Initialize query plan with defaults
            plan = QueryPlan(
                query_type=QueryType.SELECT,
                table_name=table_name
            )
            
            # Get available columns from schema 
            numeric_columns = [
                col for col in columns 
                if schema['columns'][col].get('type_category') == 'numeric'
            ]
            
            datetime_columns = [
                col for col in columns 
                if schema['columns'][col].get('type_category') == 'datetime'
            ]
            
            categorical_columns = [
                col for col in columns 
                if schema['columns'][col].get('type_category') == 'string' and
                schema['columns'][col].get('unique_count', 0) < len(schema['columns'][col].get('stats', {}).get('unique_count', 0)) * 0.5
            ]
            
            # Determine query type and extract components
            
            # Check for aggregation keywords
            for keyword, agg_func in cls.AGGREGATION_KEYWORDS.items():
                if keyword in query_lower:
                    plan.query_type = QueryType.AGGREGATE
                    
                    # Find which column to aggregate
                    target_column = cls._find_mentioned_column(query_lower, numeric_columns)
                    
                    if target_column or agg_func == 'COUNT':
                        col = target_column if target_column else '*'
                        plan.aggregations.append({
                            'function': agg_func,
                            'column': col,
                            'alias': f"{agg_func.lower()}_{col}" if col != '*' else f"{agg_func.lower()}"
                        })
                    break
            
            # Check for group by
            if "by" in query_lower or "per" in query_lower or "for each" in query_lower:
                potential_group_cols = cls._find_all_mentioned_columns(query_lower, columns)
                
                # Prefer categorical or datetime columns for grouping
                group_candidates = [
                    col for col in potential_group_cols 
                    if col in categorical_columns or col in datetime_columns
                ]
                
                if group_candidates:
                    plan.group_by_columns = group_candidates
                    if plan.query_type != QueryType.AGGREGATE:
                        plan.query_type = QueryType.GROUP_BY
            
            # Check for filtering conditions
            for keyword, operator in cls.FILTER_KEYWORDS.items():
                if keyword in query_lower:
                    filter_col = cls._find_mentioned_column(query_lower, columns)
                    if filter_col:
                        # Try to extract the value after the filter keyword
                        filter_value = cls._extract_filter_value(query_lower, keyword, filter_col)
                        
                        if filter_value is not None:
                            plan.where_conditions.append({
                                'column': filter_col,
                                'operator': operator,
                                'value': filter_value
                            })
                            if plan.query_type == QueryType.SELECT:
                                plan.query_type = QueryType.FILTER
            
            # Check for ordering
            for keyword, direction in cls.ORDER_KEYWORDS.items():
                if keyword in query_lower:
                    order_col = cls._find_mentioned_column(query_lower, columns)
                    if order_col:
                        plan.order_by.append({
                            'column': order_col,
                            'direction': direction
                        })
                        if plan.query_type == QueryType.SELECT:
                            plan.query_type = QueryType.ORDER_BY
            
            # Check for limit
            limit_patterns = [
                r'(?:show|display|get|return|limit to|top|first|limit) (\d+)',
                r'(\d+) (?:rows|results|records|entries)',
            ]
            
            for pattern in limit_patterns:
                limit_match = re.search(pattern, query_lower)
                if limit_match:
                    try:
                        plan.limit = int(limit_match.group(1))
                        if plan.query_type == QueryType.SELECT:
                            plan.query_type = QueryType.LIMIT
                        break
                    except (ValueError, IndexError):
                        pass
            
            # If no specific columns were identified, select all
            if not plan.select_columns and not plan.aggregations:
                mentioned_cols = cls._find_all_mentioned_columns(query_lower, columns)
                if mentioned_cols:
                    plan.select_columns = mentioned_cols
            
            # If the query seems complex, mark it as such
            if (len(plan.aggregations) > 1 or 
                len(plan.group_by_columns) > 1 or 
                len(plan.where_conditions) > 1 or
                plan.joins):
                plan.query_type = QueryType.COMPLEX
            
            # Log the query plan
            log_data = {k: v for k, v in plan.dict().items() if v}
            logger.info(f"Query analysis: {json.dumps(log_data)}")
            
            return plan
    
    @staticmethod
    def _find_mentioned_column(query: str, columns: List[str]) -> Optional[str]:
        """Find a column mentioned in the query."""
        for col in columns:
            col_pattern = re.compile(r'\b' + re.escape(col.lower()) + r'\b')
            if col_pattern.search(query):
                return col
        return None
    
    @staticmethod
    def _find_all_mentioned_columns(query: str, columns: List[str]) -> List[str]:
        """Find all columns mentioned in the query."""
        mentioned = []
        for col in columns:
            col_pattern = re.compile(r'\b' + re.escape(col.lower()) + r'\b')
            if col_pattern.search(query):
                mentioned.append(col)
        return mentioned
    
    @staticmethod
    def _extract_filter_value(query: str, keyword: str, column: str) -> Any:
        """Extract filter value from query based on keyword and column."""
        # Pattern to extract values after a keyword
        keyword_pattern = re.compile(r'\b' + re.escape(keyword) + r'\b\s+([^,\.]+)')
        keyword_match = keyword_pattern.search(query)
        
        if keyword_match:
            value_text = keyword_match.group(1).strip()
            
            # Try to extract a number
            number_match = re.search(r'(\d+(?:\.\d+)?)', value_text)
            if number_match:
                try:
                    return float(number_match.group(1))
                except ValueError:
                    pass
            
            # Extract quoted text or just use the text itself
            quoted_match = re.search(r'[\'"]([^\'"]+)[\'"]', value_text)
            if quoted_match:
                return quoted_match.group(1)
            else:
                # Remove the column name from the value if it's present
                value_text = re.sub(r'\b' + re.escape(column.lower()) + r'\b', '', value_text).strip()
                return value_text
        
        return None


class ResultExplainer:
    """Provides natural language explanations of query results."""
    
    @staticmethod
    def explain_results(
        sql: str, 
        results: List[Dict[str, Any]], 
        query: str, 
        plan: QueryPlan
    ) -> str:
        """
        Generate a natural language explanation of the SQL and results.
        
        Args:
            sql: SQL query
            results: Query results
            query: Original natural language query
            plan: Query plan used to generate the SQL
            
        Returns:
            Natural language explanation
        """
        with log_operation(logger, "result_explanation"):
            if not results:
                return "The query returned no results."
            
            # Explanation based on query type
            if plan.query_type == QueryType.AGGREGATE:
                return ResultExplainer._explain_aggregation(results, plan)
            elif plan.query_type == QueryType.FILTER:
                return ResultExplainer._explain_filter(results, plan)
            elif plan.query_type == QueryType.GROUP_BY:
                return ResultExplainer._explain_group_by(results, plan)
            elif plan.query_type == QueryType.ORDER_BY:
                return ResultExplainer._explain_order_by(results, plan)
            elif plan.query_type == QueryType.LIMIT:
                return ResultExplainer._explain_limit(results, plan)
            elif plan.query_type == QueryType.COMPLEX:
                return ResultExplainer._explain_complex(results, plan)
            else:
                # Generic explanation for SELECT queries
                return f"Here are {len(results)} rows matching your query."
    
    @staticmethod
    def _explain_aggregation(results: List[Dict[str, Any]], plan: QueryPlan) -> str:
        """Explain aggregation query results."""
        if not results or not plan.aggregations:
            return "The aggregation query returned no results."
        
        explanations = []
        for agg in plan.aggregations:
            func = agg.get('function', '').upper()
            col = agg.get('column', '*')
            alias = agg.get('alias', f"{func.lower()}_{col}")
            
            # Handle different aggregation functions
            if func == 'COUNT' and col == '*':
                count = results[0].get(alias, 0)
                explanations.append(f"There are {count} records in total.")
            elif func == 'COUNT':
                count = results[0].get(alias, 0)
                explanations.append(f"There are {count} records with {col}.")
            elif func == 'AVG':
                avg_value = results[0].get(alias, 0)
                explanations.append(f"The average {col} is {avg_value}.")
            elif func == 'SUM':
                sum_value = results[0].get(alias, 0)
                explanations.append(f"The sum of {col} is {sum_value}.")
            elif func == 'MAX':
                max_value = results[0].get(alias, 0)
                explanations.append(f"The maximum {col} is {max_value}.")
            elif func == 'MIN':
                min_value = results[0].get(alias, 0)
                explanations.append(f"The minimum {col} is {min_value}.")
        
        return " ".join(explanations)
    
    @staticmethod
    def _explain_filter(results: List[Dict[str, Any]], plan: QueryPlan) -> str:
        """Explain filter query results."""
        result_count = len(results)
        
        if not plan.where_conditions:
            return f"Found {result_count} records matching your criteria."
        
        condition_parts = []
        for condition in plan.where_conditions:
            col = condition.get('column', '')
            op = condition.get('operator', '=')
            val = condition.get('value', '')
            
            # Format operator for human readability
            op_readable = {
                '=': 'is equal to',
                '!=': 'is not equal to',
                '>': 'is greater than',
                '>=': 'is greater than or equal to',
                '<': 'is less than',
                '<=': 'is less than or equal to',
                'LIKE': 'contains',
                'IN': 'is in',
                'BETWEEN': 'is between'
            }.get(op, op)
            
            condition_parts.append(f"{col} {op_readable} {val}")
        
        conditions_text = " AND ".join(condition_parts)
        return f"Found {result_count} records where {conditions_text}."
    
    @staticmethod
    def _explain_group_by(results: List[Dict[str, Any]], plan: QueryPlan) -> str:
        """Explain group by query results."""
        if not plan.group_by_columns:
            return f"The query returned {len(results)} groups."
        
        group_cols = ", ".join(plan.group_by_columns)
        
        if plan.aggregations:
            # Group by with aggregation
            agg_parts = []
            for agg in plan.aggregations:
                func = agg.get('function', '').upper()
                col = agg.get('column', '*')
                agg_parts.append(f"{func} of {col}")
            
            agg_text = ", ".join(agg_parts)
            return f"The results show {agg_text} grouped by {group_cols}, with {len(results)} groups."
        else:
            # Simple group by
            return f"The results are grouped by {group_cols}, showing {len(results)} distinct groups."
    
    @staticmethod
    def _explain_order_by(results: List[Dict[str, Any]], plan: QueryPlan) -> str:
        """Explain order by query results."""
        if not plan.order_by:
            return f"The query returned {len(results)} ordered results."
        
        order_parts = []
        for order_spec in plan.order_by:
            col = order_spec.get('column', '')
            direction = order_spec.get('direction', 'ASC')
            
            direction_text = "ascending" if direction == "ASC" else "descending"
            order_parts.append(f"{col} in {direction_text} order")
        
        order_text = ", ".join(order_parts)
        return f"The query returned {len(results)} results ordered by {order_text}."
    
    @staticmethod
    def _explain_limit(results: List[Dict[str, Any]], plan: QueryPlan) -> str:
        """Explain limit query results."""
        if plan.limit is None:
            return f"The query returned {len(results)} results."
        
        return f"Showing {len(results)} records (limited to {plan.limit})."
    
    @staticmethod
    def _explain_complex(results: List[Dict[str, Any]], plan: QueryPlan) -> str:
        """Explain complex query results."""
        parts = [f"The query returned {len(results)} results."]
        
        # Add more specific details based on query components
        if plan.aggregations:
            agg_parts = []
            for agg in plan.aggregations:
                func = agg.get('function', '').upper()
                col = agg.get('column', '*')
                agg_parts.append(f"{func} of {col}")
            
            parts.append(f"It includes calculations for {', '.join(agg_parts)}.")
        
        if plan.group_by_columns:
            group_cols = ", ".join(plan.group_by_columns)
            parts.append(f"Results are grouped by {group_cols}.")
        
        if plan.where_conditions:
            parts.append(f"The data is filtered based on {len(plan.where_conditions)} conditions.")
        
        if plan.order_by:
            order_parts = []
            for order_spec in plan.order_by:
                col = order_spec.get('column', '')
                direction = order_spec.get('direction', 'ASC')
                direction_text = "ascending" if direction == "ASC" else "descending"
                order_parts.append(f"{col} ({direction_text})")
            
            parts.append(f"Results are ordered by {', '.join(order_parts)}.")
        
        return " ".join(parts)


class VisualizationSuggester:
    """Suggests appropriate visualizations based on query and results."""
    
    @staticmethod
    def suggest_visualization(
        query: str, 
        plan: QueryPlan, 
        results: List[Dict[str, Any]],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest an appropriate visualization for query results.
        
        Args:
            query: Original natural language query
            plan: Query plan used to generate the SQL
            results: Query results
            schema: Schema information for the data
            
        Returns:
            Dictionary with visualization suggestions and configuration
        """
        with log_operation(logger, "visualization_suggestion"):
            if not results:
                return {"type": None}
            
            # Convert results to DataFrame for easier analysis
            result_df = pd.DataFrame(results)
            
            # Default configuration
            vis_config = {
                "type": "table",  # Default visualization type
                "title": "",      # Chart title
                "x_axis": None,   # X-axis column
                "y_axis": None,   # Y-axis column
                "color_by": None, # Column to use for coloring
                "size_by": None,  # Column to use for sizing (scatter plots)
                "aggregation": None, # Aggregation method if applicable
                "sort": None,     # Sorting direction
                "orientation": "vertical", # Bar orientation
                "description": "Showing data in tabular format" # User-friendly description
            }
            
            # Extract important query patterns 
            query_lower = query.lower()
            
            # Set visualization title based on query
            title_match = re.search(r'(?:show|get|find|display|what is|what are)(.*?)(?:\?|$)', query_lower)
            if title_match:
                vis_config["title"] = title_match.group(1).strip().capitalize()
            else:
                # Get first 5 words of query for title
                vis_config["title"] = " ".join(query_lower.split()[:5]).capitalize()
            
            # Check for explicit visualization mentions in query
            vis_patterns = {
                "line": ["trend", "over time", "time series", "line chart", "line graph", "changes"],
                "bar": ["compare", "comparison", "bar chart", "bar graph", "ranking"],
                "histogram": ["distribution", "histogram", "spread", "frequency"],
                "scatter": ["relationship", "correlation", "scatter", "versus", "vs"],
                "pie": ["pie chart", "proportion", "percentage", "share", "breakdown"],
                "heatmap": ["heatmap", "heat map", "matrix", "grid", "intensity"],
                "box": ["box plot", "box and whisker", "range", "outliers"],
                "area": ["area chart", "area graph", "stacked area", "cumulative"],
                "radar": ["radar chart", "spider chart", "web chart"],
                "kpi": ["kpi", "key performance", "metric", "indicator", "value"]
            }
            
            for vis_type, patterns in vis_patterns.items():
                if any(pattern in query_lower for pattern in patterns):
                    vis_config["type"] = vis_type
                    vis_config["description"] = f"Showing data as a {vis_type} chart based on your request"
                    break
            
            # Get column types for all columns in the result
            categorical_cols = []
            numeric_cols = []
            temporal_cols = []
            
            for col in result_df.columns:
                if pd.api.types.is_numeric_dtype(result_df[col]):
                    numeric_cols.append(col)
                elif is_datetime_column(result_df[col]):
                    temporal_cols.append(col)
                else:
                    categorical_cols.append(col)
            
            # Recommendations based on query structure and plan
            if plan.query_type == QueryType.AGGREGATE:
                if plan.group_by_columns and len(plan.group_by_columns) == 1:
                    group_col = plan.group_by_columns[0]
                    
                    # If we have few unique values, a pie chart might be appropriate
                    if group_col in result_df.columns and len(result_df[group_col].unique()) <= 7:
                        vis_config["type"] = "pie"
                        vis_config["x_axis"] = group_col
                        vis_config["description"] = f"Showing distribution of {group_col} as a pie chart"
                    else:
                        # Bar chart for more categories
                        vis_config["type"] = "bar"
                        vis_config["x_axis"] = group_col
                        
                        # Find a numeric column for the y-axis
                        for agg in plan.aggregations:
                            alias = agg.get('alias')
                            if alias in result_df.columns and alias in numeric_cols:
                                vis_config["y_axis"] = alias
                                vis_config["description"] = f"Comparing {alias} across different {group_col} categories"
                                break
                
                elif not plan.group_by_columns:
                    # Single aggregation (e.g., COUNT(*))
                    vis_config["type"] = "kpi"
                    for agg in plan.aggregations:
                        alias = agg.get('alias')
                        if alias in result_df.columns:
                            vis_config["y_axis"] = alias
                            vis_config["description"] = f"Showing {alias} as a key metric"
                            break
            
            elif plan.query_type == QueryType.GROUP_BY:
                if len(result_df) <= 7 and len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                    vis_config["type"] = "pie"
                    vis_config["x_axis"] = categorical_cols[0]
                    vis_config["y_axis"] = numeric_cols[0]
                    vis_config["description"] = f"Showing distribution of {numeric_cols[0]} across {categorical_cols[0]} categories"
                else:
                    vis_config["type"] = "bar"
                    # Try to find appropriate x and y axes
                    if categorical_cols:
                        vis_config["x_axis"] = categorical_cols[0]
                    if numeric_cols:
                        vis_config["y_axis"] = numeric_cols[0]
                    vis_config["description"] = "Comparing values across categories"
            
            elif plan.query_type == QueryType.FILTER:
                # For filtered results, the visualization depends on the columns
                if len(temporal_cols) >= 1 and len(numeric_cols) >= 1:
                    # Time series data
                    vis_config["type"] = "line"
                    vis_config["x_axis"] = temporal_cols[0]
                    vis_config["y_axis"] = numeric_cols[0]
                    vis_config["description"] = f"Showing {numeric_cols[0]} over time"
                elif len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
                    # Categorical vs numeric
                    vis_config["type"] = "bar"
                    vis_config["x_axis"] = categorical_cols[0]
                    vis_config["y_axis"] = numeric_cols[0]
                    vis_config["description"] = f"Comparing {numeric_cols[0]} across {categorical_cols[0]} categories"
                elif len(numeric_cols) >= 2:
                    # Numeric vs numeric
                    vis_config["type"] = "scatter"
                    vis_config["x_axis"] = numeric_cols[0]
                    vis_config["y_axis"] = numeric_cols[1]
                    vis_config["description"] = f"Exploring relationship between {numeric_cols[0]} and {numeric_cols[1]}"
            
            elif plan.query_type == QueryType.ORDER_BY:
                # Ordered data is good for bar charts or tables
                if plan.order_by and len(numeric_cols) >= 1:
                    order_col = plan.order_by[0].get('column')
                    if order_col in numeric_cols:
                        vis_config["type"] = "bar"
                        vis_config["y_axis"] = order_col
                        # Find a categorical column for the x-axis
                        if categorical_cols:
                            vis_config["x_axis"] = categorical_cols[0]
                        vis_config["sort"] = plan.order_by[0].get('direction', 'ASC')
                        vis_config["description"] = f"Showing {order_col} in {'ascending' if vis_config['sort'] == 'ASC' else 'descending'} order"
                    else:
                        vis_config["type"] = "table"
                        vis_config["description"] = "Showing sorted data in tabular format"
            
            # Analyze data patterns for additional insights
            # Check for correlation between numeric columns
            if len(numeric_cols) >= 2 and len(result_df) >= 5:
                try:
                    # Sample correlation calculation
                    corr = result_df[numeric_cols[:2]].corr().iloc[0, 1]
                    if abs(corr) > 0.7:  # Strong correlation
                        vis_config["type"] = "scatter"
                        vis_config["x_axis"] = numeric_cols[0]
                        vis_config["y_axis"] = numeric_cols[1]
                        vis_config["description"] = f"Showing strong {'positive' if corr > 0 else 'negative'} correlation between {numeric_cols[0]} and {numeric_cols[1]}"
                except:
                    # Correlation calculation might fail for various reasons
                    pass
                
            # Check if we have a single KPI
            if len(result_df) == 1 and len(numeric_cols) == 1:
                vis_config["type"] = "kpi"
                vis_config["y_axis"] = numeric_cols[0]
                vis_config["description"] = f"Showing {numeric_cols[0]} as a key metric"
            
            # If we have geographical data, consider a map
            geo_columns = [col for col in result_df.columns if any(geo_term in col.lower() for geo_term in 
                          ["country", "state", "city", "region", "province", "county", "latitude", "longitude", "zip", "postal"])]
            
            if geo_columns and any(col in result_df.columns for col in ["latitude", "longitude"]):
                vis_config["type"] = "map"
                vis_config["x_axis"] = next((col for col in result_df.columns if "longitude" in col.lower()), None)
                vis_config["y_axis"] = next((col for col in result_df.columns if "latitude" in col.lower()), None)
                vis_config["description"] = "Displaying geographical data on a map"
            
            # For large result sets, prefer tables
            if len(result_df) > 100 and vis_config["type"] not in ["kpi", "line", "scatter"]:
                vis_config["type"] = "table"
                vis_config["description"] = "Showing data in tabular format due to large result set"
            
            # Add smart recommendations
            if vis_config["type"] == "bar" and vis_config["x_axis"] and len(result_df[vis_config["x_axis"]].unique()) > 15:
                # Too many categories for a bar chart
                vis_config["recommendations"] = [
                    {"type": "table", "reason": "Many categories might be better viewed in a table"},
                    {"type": "bar", "reason": "Consider grouping some categories for a clearer visualization"}
                ]
            
            # Log the visualization suggestion
            logger.info(f"Suggesting visualization: {vis_config['type']} for query result")
            
            return vis_config


class DataInsightGenerator:
    """Generates insights about the data in the query results."""
    
    @staticmethod
    def generate_insights(
        results: List[Dict[str, Any]], 
        plan: QueryPlan,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate insights about the data in the query results.
        
        Args:
            results: Query results
            plan: Query plan used to generate the SQL
            schema: Schema information for the data
            
        Returns:
            Dictionary with insights
        """
        with log_operation(logger, "insight_generation"):
            if not results:
                return {}
            
            insights = {}
            
            # Convert results to DataFrame for analysis
            result_df = pd.DataFrame(results)
            
            # Get numeric columns
            numeric_cols = [
                col for col in result_df.columns 
                if pd.api.types.is_numeric_dtype(result_df[col])
            ]
            
            # Basic statistics for numeric columns
            for col in numeric_cols:
                if col in result_df.columns:
                    insights[col] = {
                        "min": result_df[col].min() if not result_df[col].empty else None,
                        "max": result_df[col].max() if not result_df[col].empty else None,
                        "mean": result_df[col].mean() if not result_df[col].empty else None,
                        "median": result_df[col].median() if not result_df[col].empty else None
                    }
            
            # Additional insights based on query type
            if plan.query_type == QueryType.AGGREGATE and plan.aggregations:
                # For aggregation queries, we can add insights about the aggregated values
                for agg in plan.aggregations:
                    func = agg.get('function', '').upper()
                    col = agg.get('column', '*')
                    alias = agg.get('alias', f"{func.lower()}_{col}")
                    
                    if alias in result_df.columns:
                        insights["aggregation"] = {
                            "function": func,
                            "column": col,
                            "value": result_df[alias].iloc[0] if len(result_df) > 0 else None
                        }
            
            elif plan.query_type == QueryType.GROUP_BY and plan.group_by_columns:
                # For group by queries, we can add insights about the distribution
                group_col = plan.group_by_columns[0] if plan.group_by_columns else None
                
                if group_col and group_col in result_df.columns:
                    value_counts = result_df[group_col].value_counts()
                    
                    insights["distribution"] = {
                        "column": group_col,
                        "most_common": value_counts.index[0] if not value_counts.empty else None,
                        "most_common_count": value_counts.iloc[0] if not value_counts.empty else None,
                        "unique_values": len(value_counts)
                    }
            
            return insights


def data_chat_agent(user_query: UserQuery) -> QueryResponse:
    """
    Process a user query and generate a response.
    
    Args:
        user_query: User query with context
        
    Returns:
        Structured response
    """
    with log_operation(logger, "data_chat_agent", log_success=True):
        with mdc_context(query=user_query.query):
            # Extract schema information from context
            schema = user_query.context.get("schema", {})
            data_sample = user_query.context.get("data_sample", [])
            
            # Create a sample DataFrame from the data sample
            sample_df = pd.DataFrame(data_sample) if data_sample else pd.DataFrame()
            
            start_time = time.time()
            
            try:
                # Analyze the query to create a structured plan
                query_plan = NLQueryAnalyzer.analyze_query(user_query.query, schema)
                
                # Generate SQL from the plan
                sql = SQLGenerator.generate_sql(query_plan)
                
                # In a real implementation, we'd execute the SQL against actual data
                # For demo purposes, just return a sample from the provided data
                if not sample_df.empty:
                    results = sample_df.head(10).to_dict(orient="records")
                else:
                    results = []
                
                query_time_ms = int((time.time() - start_time) * 1000)
                
                # Generate explanation for the results
                explanation = ResultExplainer.explain_results(
                    sql, results, user_query.query, query_plan
                )
                
                # Suggest visualization type
                visualization_type = VisualizationSuggester.suggest_visualization(
                    user_query.query, query_plan, results, schema
                )
                
                # Generate insights about the data
                column_insights = DataInsightGenerator.generate_insights(
                    results, query_plan, schema
                )
                
                # Create response
                response = QueryResponse(
                    sql=sql,
                    result=results,
                    explanation=explanation,
                    visualization_type=visualization_type["type"],
                    query_time_ms=query_time_ms,
                    row_count=len(results),
                    column_insights=column_insights
                )
                
                logger.info(f"Query processed successfully in {query_time_ms}ms")
                return response
                
            except Exception as e:
                logger.exception(f"Error processing query: {str(e)}")
                
                # Return a fallback response with an error message
                return QueryResponse(
                    sql="SELECT * FROM data LIMIT 5",
                    result=sample_df.head(5).to_dict(orient="records") if not sample_df.empty else [],
                    explanation=f"I encountered an error processing your query: {str(e)}. Here's a sample of the data instead.",
                    visualization_type="table",
                    query_time_ms=int((time.time() - start_time) * 1000),
                    row_count=min(5, len(sample_df)) if not sample_df.empty else 0
                ) 

def is_datetime_column(series: pd.Series) -> bool:
    """Check if a pandas Series likely contains datetime data."""
    # Check if it's already a datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    
    # If it's a string series, check if it looks like dates
    if pd.api.types.is_string_dtype(series):
        # Sample the first few non-null values
        sample = series.dropna().head(5)
        if len(sample) == 0:
            return False
        
        # Check if they match common date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
            r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
        ]
        
        for val in sample:
            if any(re.match(pattern, str(val)) for pattern in date_patterns):
                return True
    
    return False 