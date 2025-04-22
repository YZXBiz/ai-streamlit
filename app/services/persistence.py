"""
Persistence service for data storage.

This module handles session persistence and MongoDB operations.
"""
import os
import json
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional, List

# Conditionally import MongoDB
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False

class PersistenceService:
    """Service for data persistence operations."""
    
    def __init__(self):
        """Initialize the persistence service."""
        self.environment = os.environ.get("ENVIRONMENT", "development")
        self.mongodb_uri = os.environ.get("MONGODB_URI")
        self.client = None
        self.db = None
        
        # Connect to MongoDB if available
        if MONGODB_AVAILABLE and self.mongodb_uri:
            try:
                self.client = MongoClient(self.mongodb_uri)
                self.db = self.client.chatbot_db
            except Exception as e:
                st.warning(f"Could not connect to MongoDB: {e}")
                st.warning("Persistence will be limited to session storage only.")
    
    def save_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Save or update session data.
        
        Args:
            session_data: Session data to save
            
        Returns:
            Success flag
        """
        session_id = session_data.get("session_id", str(uuid.uuid4()))
        session_data["last_active"] = datetime.now()
        
        # Store in MongoDB if available
        if self.db:
            try:
                self.db.sessions.update_one(
                    {"session_id": session_id},
                    {"$set": session_data},
                    upsert=True
                )
                return True
            except Exception as e:
                st.error(f"Error saving to MongoDB: {e}")
                return self._save_local(session_data, "sessions", session_id)
        else:
            # Fallback to local storage
            return self._save_local(session_data, "sessions", session_id)
    
    def save_dataset_metadata(self, session_id: str, dataset_info: Dict[str, Any]) -> bool:
        """
        Save dataset metadata linked to a session.
        
        Args:
            session_id: Session identifier
            dataset_info: Dataset metadata
            
        Returns:
            Success flag
        """
        dataset_info["timestamp"] = datetime.now()
        
        # Store in MongoDB if available
        if self.db:
            try:
                self.db.sessions.update_one(
                    {"session_id": session_id},
                    {"$push": {"datasets": dataset_info}}
                )
                return True
            except Exception as e:
                st.error(f"Error saving dataset metadata: {e}")
                return False
        else:
            # Add to session state
            if "datasets" not in st.session_state:
                st.session_state.datasets = []
            
            st.session_state.datasets.append(dataset_info)
            return True
    
    def save_query_result(self, session_id: str, query_data: Dict[str, Any]) -> bool:
        """
        Save query and result information.
        
        Args:
            session_id: Session identifier
            query_data: Query and result data
            
        Returns:
            Success flag
        """
        query_data["timestamp"] = datetime.now()
        query_id = query_data.get("query_id", str(uuid.uuid4()))
        query_data["query_id"] = query_id
        
        # Store in MongoDB if available
        if self.db:
            try:
                # Add to session's query history
                self.db.sessions.update_one(
                    {"session_id": session_id},
                    {"$push": {"queries": query_data}}
                )
                
                # Store in queries collection for analytics
                self.db.queries.insert_one({
                    "session_id": session_id,
                    **query_data
                })
                return True
            except Exception as e:
                st.error(f"Error saving query result: {e}")
                return self._save_local(query_data, "queries", query_id)
        else:
            # Add to session state
            if "query_history" not in st.session_state:
                st.session_state.query_history = []
            
            st.session_state.query_history.append(query_data)
            return True
    
    def export_session_data(self, session_id: str, format: str = "json") -> Optional[str]:
        """
        Export all session data to the specified format.
        
        Args:
            session_id: Session identifier
            format: Output format (json, csv)
            
        Returns:
            Exported data as string or None if error
        """
        try:
            # Get session data
            session_data = self._get_session_data(session_id)
            
            if not session_data:
                return None
            
            # Format output based on requested format
            if format == "json":
                return json.dumps(session_data, default=str, indent=2)
            elif format == "csv":
                # This is a simplification - real implementation would need more work
                # to flatten the hierarchical session data
                return pd.DataFrame([session_data]).to_csv(index=False)
            else:
                st.error(f"Unsupported export format: {format}")
                return None
        except Exception as e:
            st.error(f"Error exporting session data: {e}")
            return None
    
    def _get_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete session data.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session data dictionary or None if not found
        """
        # Try to get from MongoDB first
        if self.db:
            try:
                session_data = self.db.sessions.find_one({"session_id": session_id})
                if session_data:
                    # Convert ObjectId to string for serialization
                    session_data["_id"] = str(session_data["_id"])
                    return session_data
            except Exception:
                pass
        
        # Fall back to session state
        if "session_id" in st.session_state and st.session_state.session_id == session_id:
            # Build session data from various state components
            session_data = {
                "session_id": session_id,
                "created_at": st.session_state.get("created_at", datetime.now()),
                "datasets": st.session_state.get("datasets", []),
                "query_history": st.session_state.get("query_history", []),
                "messages": st.session_state.get("messages", [])
            }
            return session_data
        
        return None
    
    def _save_local(self, data: Dict[str, Any], collection: str, item_id: str) -> bool:
        """
        Save data locally when MongoDB is not available.
        
        Args:
            data: Data to save
            collection: Type of data (sessions, queries, etc.)
            item_id: Unique identifier for the item
            
        Returns:
            Success flag
        """
        try:
            # In development, we can store to a local JSON file
            if self.environment == "development":
                # Create data directory if it doesn't exist
                os.makedirs("data", exist_ok=True)
                
                # Determine the file path
                file_path = f"data/{collection}_{item_id}.json"
                
                # Write data to file
                with open(file_path, "w") as f:
                    json.dump(data, f, default=str, indent=2)
            
            # Add to session state (simplified)
            collection_key = f"{collection}_data"
            if collection_key not in st.session_state:
                st.session_state[collection_key] = {}
            
            st.session_state[collection_key][item_id] = data
            
            return True
        except Exception as e:
            st.error(f"Error saving data locally: {e}")
            return False 