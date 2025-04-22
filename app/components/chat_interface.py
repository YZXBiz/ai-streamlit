"""
Chat interface component.

This module handles the chat UI and interactions with the query service.
"""
import streamlit as st
import pandas as pd
import time
from typing import List, Dict, Any

from services.query_service import QueryService

def render_chat_interface():
    """Render the chat interface for data interactions."""
    st.header("Chat with Your Data")
    
    query_service = QueryService()
    
    # Display chat messages from history
    display_chat_history()
    
    # React to user input
    if prompt := st.chat_input("What would you like to know about your data?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown("Thinking...")
            
            try:
                # Display a spinner while processing
                with st.spinner("Generating response..."):
                    # Generate SQL and response
                    query_result = query_service.process_query(prompt, st.session_state.data)
                    
                    if query_result:
                        # Progressive response display
                        response_parts = [
                            f"I've analyzed your data and here's what I found:\n\n",
                            f"{query_result['explanation']}\n\n"
                        ]
                        
                        # Show SQL if available
                        if 'sql' in query_result:
                            response_parts.append(f"SQL Query:\n```sql\n{query_result['sql']}\n```\n\n")
                        
                        # Show visualization suggestion if available
                        if 'visualization_type' in query_result and query_result['visualization_type']:
                            response_parts.append(f"This data would look good as a {query_result['visualization_type']} chart.")
                        
                        # Progressive display
                        full_response = ""
                        for part in response_parts:
                            full_response += part
                            response_placeholder.markdown(full_response)
                            time.sleep(0.1)
                        
                        # Display result data if available
                        if 'result_data' in query_result and not query_result['result_data'].empty:
                            st.dataframe(query_result['result_data'])
                            
                            # Add download button for results
                            csv = query_result['result_data'].to_csv(index=False)
                            st.download_button(
                                label="Download results as CSV",
                                data=csv,
                                file_name="query_result.csv",
                                mime="text/csv",
                            )
                    else:
                        response_placeholder.markdown("I couldn't find an answer to your question. Please try rephrasing or ask something else about your data.")
            
            except Exception as e:
                response_placeholder.markdown(f"Sorry, I encountered an error: {str(e)}")
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_placeholder.markdown if 'query_result' in locals() else "I couldn't process your request."
        })
        
        # Update the UI to show the new message
        st.experimental_rerun()

def display_chat_history():
    """Display the chat history from session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def add_message_feedback(message_idx: int):
    """
    Add feedback options to a message.
    
    Args:
        message_idx: Index of the message in the chat history
    """
    col1, col2 = st.columns([1, 10])
    
    with col1:
        if st.button("👍", key=f"thumbs_up_{message_idx}"):
            st.session_state.messages[message_idx]["feedback"] = "positive"
            st.experimental_rerun()
            
        if st.button("👎", key=f"thumbs_down_{message_idx}"):
            st.session_state.messages[message_idx]["feedback"] = "negative"
            st.experimental_rerun()
            
    # If negative feedback, show text area for additional feedback
    if st.session_state.messages[message_idx].get("feedback") == "negative":
        feedback_text = st.text_area(
            "What went wrong? (Optional)",
            key=f"feedback_text_{message_idx}"
        )
        
        if st.button("Submit Feedback", key=f"submit_feedback_{message_idx}"):
            st.session_state.messages[message_idx]["feedback_text"] = feedback_text
            st.success("Thank you for your feedback!")
            st.experimental_rerun() 