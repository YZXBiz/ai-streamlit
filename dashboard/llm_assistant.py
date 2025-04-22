"""LLM integration for the clustering dashboard.

This module provides LLM-powered chat capabilities for the dashboard to analyze
clustering results and provide insights through natural language.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


class ClusteringAssistant:
    """LLM-powered assistant for cluster analysis.
    
    This class handles interactions with an LLM service to analyze
    clustering results and provide insights through a chat interface.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the clustering assistant.
        
        Args:
            api_key: Optional API key for the LLM service. If not provided,
                will attempt to use an environment variable.
        """
        # Use provided API key or check environment variables
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.cluster_data = None
        self.model = "gpt-3.5-turbo"  # Default model
        
    def load_data(self, df: pd.DataFrame) -> None:
        """Load clustering data for the assistant to reference.
        
        Args:
            df: DataFrame containing clustering results
        """
        self.cluster_data = df
        
    def get_cluster_summary(self) -> str:
        """Generate a summary of the loaded cluster data.
        
        Returns:
            A text summary of the clustering results
        """
        if self.cluster_data is None:
            return "No cluster data has been loaded yet."
            
        # Basic stats about the clusters
        n_clusters = self.cluster_data["CLUSTER"].nunique()
        n_stores = len(self.cluster_data)
        
        summary = f"Dataset contains {n_stores} stores grouped into {n_clusters} clusters.\n\n"
        
        # Add cluster distribution
        summary += "Cluster distribution:\n"
        cluster_counts = self.cluster_data["CLUSTER"].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            summary += f"- Cluster {cluster}: {count} stores ({count/n_stores:.1%})\n"
            
        return summary
    
    def get_response(self, user_query: str) -> str:
        """Get a response from the LLM based on the user's query.
        
        Args:
            user_query: The user's question about clustering results
            
        Returns:
            The LLM response
        """
        # If OpenAI is configured, you would call the API here
        if self.api_key:
            try:
                # This is where you'd implement the actual OpenAI integration
                # For now, we'll return a placeholder response with data references
                return self._generate_mock_response(user_query)
            except Exception as e:
                return f"Error connecting to LLM service: {str(e)}"
        else:
            return (
                "LLM service not configured. Please set the OPENAI_API_KEY "
                "environment variable or provide an API key on initialization."
            )
    
    def _generate_mock_response(self, query: str) -> str:
        """Generate a mock response for demonstration purposes.
        
        Args:
            query: The user's question
            
        Returns:
            A mock response that simulates what an LLM would return
        """
        query_lower = query.lower()
        
        # Check if we have cluster data
        if self.cluster_data is None:
            return (
                "I don't see any cluster data loaded yet. Please upload your "
                "clustering results first so I can provide insights."
            )
        
        # Different mock responses based on query topics
        if "what" in query_lower and "cluster" in query_lower:
            return (
                "Based on the clustering results, I can see several distinct store segments:\n\n"
                "- **Cluster 0**: High-volume urban stores with above-average transaction sizes\n"
                "- **Cluster 1**: Suburban stores with moderate traffic and high loyalty\n"
                "- **Cluster 2**: Small format stores with high frequency, lower basket size\n\n"
                "These clusters differ mainly in their sales volume, transaction patterns, and "
                "customer demographics."
            )
        elif "how" in query_lower and "improv" in query_lower:
            return (
                "To improve performance across your store clusters, you could consider:\n\n"
                "1. **For Cluster 0 (Urban high-volume)**: Focus on premium products and express checkout\n"
                "2. **For Cluster 1 (Suburban)**: Enhance loyalty programs and family-oriented merchandise\n"
                "3. **For Cluster 2 (Small format)**: Optimize for quick trips and essential items\n\n"
                "Each cluster has different customer needs, so tailoring operations and merchandising "
                "accordingly will yield the best results."
            )
        elif "which" in query_lower and ("feature" in query_lower or "variable" in query_lower):
            return (
                "The most important features driving the cluster formation are:\n\n"
                "1. **Transaction Count** (importance: 0.32)\n"
                "2. **Average Basket Size** (importance: 0.28)\n"
                "3. **Store Square Footage** (importance: 0.15)\n"
                "4. **Weekend to Weekday Ratio** (importance: 0.12)\n\n"
                "Transaction patterns and store size appear to be the dominant factors in "
                "distinguishing between store clusters."
            )
        else:
            return (
                "That's an interesting question about your store clusters. In a full implementation, "
                "I would analyze your specific clustering data to provide a detailed answer. "
                "For now, you can try asking about:\n\n"
                "- What the different clusters represent\n"
                "- Which features are most important for the clustering\n"
                "- How to improve performance for specific clusters\n"
                "- Comparing characteristics between clusters"
            )


def display_chat_interface():
    """Display a Streamlit chat interface with LLM-powered responses."""
    # Initialize the assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = ClusteringAssistant()
    
    # Get assistant instance
    assistant = st.session_state.assistant
    
    # If there's cluster data in the session, load it
    if "cluster_data" in st.session_state:
        if assistant.cluster_data is None:
            assistant.load_data(st.session_state.cluster_data)
            st.info("Loaded cluster data for AI analysis")
    
    # Cluster data summary
    if assistant.cluster_data is not None:
        with st.expander("Cluster Data Summary", expanded=False):
            st.markdown(assistant.get_cluster_summary())
    
    # Initialize chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your Clustering AI Assistant. How can I help you understand your store clusters today?"}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input for new message
    if prompt := st.chat_input("Ask about your clusters..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = assistant.get_response(prompt)
            message_placeholder.markdown(response)
            
            # Add feedback buttons (thumbs up/down)
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                st.button("👍", key=f"thumbs_up_{len(st.session_state.messages)}")
            with col2:
                st.button("👎", key=f"thumbs_down_{len(st.session_state.messages)}")
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


def process_chat_message(message: str, df: pd.DataFrame) -> str:
    """Process a chat message and return a response based on the data.
    
    This function is used by the chat_interface component to handle
    user messages and generate responses using the assistant.
    
    Args:
        message: The user's message/query
        df: The DataFrame containing the data to analyze
        
    Returns:
        The assistant's response to the query
    """
    # Initialize the assistant if not already done
    if "assistant" not in st.session_state:
        st.session_state.assistant = ClusteringAssistant()
    
    # Get assistant instance
    assistant = st.session_state.assistant
    
    # Load the data if needed
    assistant.load_data(df)
    
    # Get the response
    response = assistant.get_response(message)
    
    return response 