"""
Chat interface component for the PandasAI Streamlit application.

This module provides the chat interface for interacting with the PandasAI backend.
"""

import pandas as pd
import pandasai as pai
import streamlit as st

from frontend.utils.auth import authenticated_request


def render_chat_interface():
    """
    Render the chat interface for interacting with PandasAI.

    This function displays:
    - Chat history
    - Dataframe selection (if multiple dataframes are loaded)
    - Chat input for asking questions
    """
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get available dataframes
    available_dfs = st.session_state.loaded_dataframes

    # Display dataframe selection if multiple dataframes are loaded
    selected_session = None
    if len(available_dfs) > 1:
        # Get active chat sessions from the backend
        response = authenticated_request("get", "/sessions")
        if response and response.status_code == 200:
            sessions = response.json()
            if sessions:
                session_options = {f"{s['name']} (ID: {s['id']})": s["id"] for s in sessions}
                selected_session_name = st.selectbox(
                    "Select a chat session:", options=list(session_options.keys()), index=0
                )
                if selected_session_name:
                    selected_session = session_options[selected_session_name]

        selected_dfs = st.multiselect(
            "Select datasets to query (leave empty to query all):", available_dfs, default=[]
        )
    else:
        selected_dfs = available_dfs

    # Accept user input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from PandasAI
        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                try:
                    # Send the message to the backend API
                    if selected_session:
                        # Use the backend API for chat
                        response = authenticated_request(
                            "post",
                            f"/sessions/{selected_session}/messages",
                            json={"content": prompt},
                        )

                        if response and response.status_code in (200, 201):
                            messages = response.json()
                            # The API returns both user and assistant messages
                            # We only need the assistant message (the second one)
                            if len(messages) > 1:
                                ai_message = messages[1]
                                response_content = ai_message["content"]
                                st.markdown(response_content)

                                # Add assistant response to chat history
                                st.session_state.messages.append(
                                    {"role": "assistant", "content": response_content}
                                )
                            else:
                                st.error("Unexpected response format from API")
                        else:
                            error_msg = "Error getting response from API"
                            if response:
                                error_msg = response.json().get("detail", error_msg)
                            st.error(error_msg)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": f"Error: {error_msg}"}
                            )
                    else:
                        # Fallback to direct PandasAI processing
                        # If no specific dataframes are selected, use all available
                        if not selected_dfs and len(available_dfs) > 1:
                            selected_dfs = available_dfs

                        # Get the dataframes to analyze
                        dataframes = [
                            st.session_state.analyzer.dataframe_manager.get_dataframe(df_name)
                            for df_name in selected_dfs
                        ]

                        # If only one dataframe, use direct chat method
                        if len(dataframes) == 1:
                            response_obj = dataframes[0].chat(prompt)
                        # If multiple dataframes, create a collection
                        elif len(dataframes) > 1:
                            # Create a collection for cross-dataframe analysis
                            collection = pai.Collection(dataframes)
                            response_obj = collection.chat(prompt)

                        # Process the response based on its type
                        if isinstance(response_obj, str):
                            response = response_obj
                            st.markdown(response)
                        elif isinstance(response_obj, pd.DataFrame):
                            response = f"Here's the result:\n\n"
                            st.markdown(response)
                            st.dataframe(response_obj)
                        elif hasattr(response_obj, "figure"):
                            response = "Here's the visualization:"
                            st.markdown(response)
                            st.pyplot(response_obj.figure)
                        else:
                            response = str(response_obj)
                            st.markdown(response)

                        # Get and display the generated code if available
                        if hasattr(st.session_state.analyzer, "get_last_code"):
                            code = st.session_state.analyzer.get_last_code()
                            if code:
                                with st.expander("View Generated Code"):
                                    st.code(code, language="python")

                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error analyzing data: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
