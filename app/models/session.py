"""
Session model for managing session state.

This module defines the session state structure and operations.
"""
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DatasetMetadata(BaseModel):
    """Model for dataset metadata."""
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source_type: str  # "file", "snowflake", etc.
    table_name: str
    row_count: int
    column_count: int
    columns: List[str]
    created_at: datetime = Field(default_factory=datetime.now)
    additional_info: Dict[str, Any] = Field(default_factory=dict)

class QueryMetadata(BaseModel):
    """Model for query metadata."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    natural_language: str
    sql: str
    timestamp: datetime = Field(default_factory=datetime.now)
    execution_time_ms: Optional[int] = None
    result_count: Optional[int] = None
    visualization_type: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None

class Message(BaseModel):
    """Model for chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    feedback: Optional[Dict[str, Any]] = None

class VisualizationState(BaseModel):
    """Model for visualization state."""
    chart_type: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)
    aggregations: List[Dict[str, Any]] = Field(default_factory=list)

class SessionState(BaseModel):
    """Model for complete session state."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    last_active: datetime = Field(default_factory=datetime.now)
    datasets: List[DatasetMetadata] = Field(default_factory=list)
    queries: List[QueryMetadata] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)
    visualization_state: VisualizationState = Field(default_factory=VisualizationState)
    
    def update_last_active(self):
        """Update the last active timestamp."""
        self.last_active = datetime.now()
    
    def add_dataset(self, dataset_metadata: DatasetMetadata):
        """
        Add a dataset to the session.
        
        Args:
            dataset_metadata: Dataset metadata to add
        """
        self.datasets.append(dataset_metadata)
        self.update_last_active()
    
    def add_query(self, query_metadata: QueryMetadata):
        """
        Add a query to the session.
        
        Args:
            query_metadata: Query metadata to add
        """
        self.queries.append(query_metadata)
        self.update_last_active()
    
    def add_message(self, role: str, content: str):
        """
        Add a message to the session.
        
        Args:
            role: Message role ("user" or "assistant")
            content: Message content
        """
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.update_last_active()
    
    def update_visualization_state(self, chart_type: Optional[str] = None, 
                                  filters: Optional[Dict[str, Any]] = None,
                                  aggregations: Optional[List[Dict[str, Any]]] = None):
        """
        Update the visualization state.
        
        Args:
            chart_type: The chart type
            filters: Filters applied to the visualization
            aggregations: Aggregations applied to the visualization
        """
        if chart_type:
            self.visualization_state.chart_type = chart_type
        
        if filters:
            self.visualization_state.filters = filters
        
        if aggregations:
            self.visualization_state.aggregations = aggregations
        
        self.update_last_active()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session state to dictionary.
        
        Returns:
            Dictionary representation of session state
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "datasets": [dataset.dict() for dataset in self.datasets],
            "queries": [query.dict() for query in self.queries],
            "messages": [message.dict() for message in self.messages],
            "visualization_state": self.visualization_state.dict()
        } 