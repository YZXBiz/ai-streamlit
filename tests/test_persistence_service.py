"""Tests for the persistence service functionality."""
import unittest
from unittest.mock import patch, MagicMock, ANY
import pandas as pd
import streamlit as st
import json
import uuid
from datetime import datetime

from app.services.persistence import PersistenceService


class TestPersistenceService(unittest.TestCase):
    """Test cases for PersistenceService."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock streamlit session state
        if not hasattr(st, 'session_state'):
            st.session_state = {}
        
        # Create test session data
        self.test_session_id = str(uuid.uuid4())
        self.test_session_data = {
            "session_id": self.test_session_id,
            "user_id": "test_user",
            "created_at": datetime.now(),
            "datasets": [],
            "queries": []
        }
        
        # Create test query data
        self.test_query_data = {
            "query": "Show me all products",
            "sql": "SELECT * FROM products",
            "success": True,
            "timestamp": datetime.now()
        }
        
        # Create test dataset info
        self.test_dataset_info = {
            "name": "test_dataset.csv",
            "rows": 100,
            "columns": 5,
            "size_bytes": 10240,
            "timestamp": datetime.now()
        }

    @patch('app.services.persistence.MongoClient')
    def test_mongodb_initialization(self, mock_mongo_client):
        """Test MongoDB client initialization."""
        # Mock environment variables
        with patch.dict('os.environ', {'MONGODB_URI': 'mongodb://localhost:27017/'}):
            # Create service
            service = PersistenceService()
            
            # Verify MongoDB client was initialized
            mock_mongo_client.assert_called_once_with('mongodb://localhost:27017/')
            
            # Verify database reference
            self.assertIsNotNone(service.db)
            self.assertEqual(service.client, mock_mongo_client.return_value)
            self.assertEqual(service.db, mock_mongo_client.return_value.chatbot_db)
    
    @patch('app.services.persistence.MongoClient')
    def test_mongodb_connection_error(self, mock_mongo_client):
        """Test handling of MongoDB connection errors."""
        # Mock MongoDB client to raise exception
        mock_mongo_client.side_effect = Exception("Connection error")
        
        # Mock environment variables
        with patch.dict('os.environ', {'MONGODB_URI': 'mongodb://localhost:27017/'}):
            # Mock streamlit warning
            with patch('streamlit.warning') as mock_warning:
                # Create service
                service = PersistenceService()
                
                # Verify warnings were shown
                mock_warning.assert_called()
                self.assertIsNone(service.client)
                self.assertIsNone(service.db)
    
    @patch('app.services.persistence.PersistenceService._save_local')
    def test_save_session_without_mongodb(self, mock_save_local):
        """Test saving session when MongoDB is not available."""
        # Setup
        service = PersistenceService()
        service.db = None  # Ensure MongoDB is not available
        mock_save_local.return_value = True
        
        # Call method
        result = service.save_session(self.test_session_data)
        
        # Verify results
        self.assertTrue(result)
        mock_save_local.assert_called_once_with(
            self.test_session_data, 
            "sessions", 
            self.test_session_id
        )
    
    @patch('app.services.persistence.MongoClient')
    def test_save_session_with_mongodb(self, mock_mongo_client):
        """Test saving session with MongoDB available."""
        # Setup mock database and collection
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_db.sessions = mock_collection
        
        mock_client = MagicMock()
        mock_client.chatbot_db = mock_db
        mock_mongo_client.return_value = mock_client
        
        # Mock environment variables
        with patch.dict('os.environ', {'MONGODB_URI': 'mongodb://localhost:27017/'}):
            # Create service
            service = PersistenceService()
            
            # Call method
            result = service.save_session(self.test_session_data)
            
            # Verify results
            self.assertTrue(result)
            mock_collection.update_one.assert_called_once_with(
                {"session_id": self.test_session_id},
                {"$set": ANY},
                upsert=True
            )
    
    def test_save_dataset_metadata_without_mongodb(self):
        """Test saving dataset metadata without MongoDB."""
        # Setup
        service = PersistenceService()
        service.db = None  # Ensure MongoDB is not available
        
        # Call method
        result = service.save_dataset_metadata(
            self.test_session_id, 
            self.test_dataset_info
        )
        
        # Verify results
        self.assertTrue(result)
        self.assertIn("datasets", st.session_state)
        self.assertEqual(len(st.session_state.datasets), 1)
        self.assertEqual(
            st.session_state.datasets[0]["name"], 
            self.test_dataset_info["name"]
        )
    
    @patch('app.services.persistence.MongoClient')
    def test_save_query_result_with_mongodb(self, mock_mongo_client):
        """Test saving query result with MongoDB available."""
        # Setup mock database and collections
        mock_db = MagicMock()
        mock_sessions = MagicMock()
        mock_queries = MagicMock()
        mock_db.sessions = mock_sessions
        mock_db.queries = mock_queries
        
        mock_client = MagicMock()
        mock_client.chatbot_db = mock_db
        mock_mongo_client.return_value = mock_client
        
        # Mock environment variables
        with patch.dict('os.environ', {'MONGODB_URI': 'mongodb://localhost:27017/'}):
            # Create service
            service = PersistenceService()
            
            # Call method
            result = service.save_query_result(
                self.test_session_id, 
                self.test_query_data
            )
            
            # Verify results
            self.assertTrue(result)
            mock_sessions.update_one.assert_called_once()
            mock_queries.insert_one.assert_called_once()
    
    def test_export_session_data_json(self):
        """Test exporting session data as JSON."""
        service = PersistenceService()
        
        # Mock _get_session_data
        with patch.object(
            service, '_get_session_data', return_value=self.test_session_data
        ) as mock_get_data:
            # Call method
            result = service.export_session_data(self.test_session_id, format="json")
            
            # Verify results
            self.assertIsNotNone(result)
            # Parse the JSON to verify it's valid
            parsed = json.loads(result)
            self.assertEqual(parsed["session_id"], self.test_session_id)
            
            # Verify method calls
            mock_get_data.assert_called_once_with(self.test_session_id)
    
    def test_export_session_data_csv(self):
        """Test exporting session data as CSV."""
        service = PersistenceService()
        
        # Mock _get_session_data
        with patch.object(
            service, '_get_session_data', return_value=self.test_session_data
        ) as mock_get_data:
            # Call method
            result = service.export_session_data(self.test_session_id, format="csv")
            
            # Verify results
            self.assertIsNotNone(result)
            # Verify it looks like CSV (has commas and the session_id)
            self.assertIn(",", result)
            self.assertIn(self.test_session_id, result)
            
            # Verify method calls
            mock_get_data.assert_called_once_with(self.test_session_id)
    
    def test_export_session_data_unsupported_format(self):
        """Test exporting session data with unsupported format."""
        service = PersistenceService()
        
        # Mock _get_session_data and streamlit error
        with patch.object(
            service, '_get_session_data', return_value=self.test_session_data
        ) as mock_get_data:
            with patch('streamlit.error') as mock_error:
                # Call method
                result = service.export_session_data(
                    self.test_session_id, format="unsupported"
                )
                
                # Verify results
                self.assertIsNone(result)
                mock_error.assert_called_once()
                mock_get_data.assert_called_once_with(self.test_session_id)


if __name__ == '__main__':
    unittest.main() 