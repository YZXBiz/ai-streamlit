import streamlit as st

from app.utils.data_utils import initialize_agent, load_dataframe


def file_uploader():
    """Render the file upload interface and process uploaded files."""
    st.subheader("Upload Data")

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # Load the dataframe
        df, error = load_dataframe(uploaded_file)

        if error:
            st.error(f"Error loading file: {error}")
            return

        # Get the API key (from env or user input)
        api_key = st.session_state.get("api_key", None)
        if not api_key:
            api_key = st.text_input("Enter your OpenAI API Key:", type="password")
            if not api_key:
                st.warning("Please enter an OpenAI API key to continue")
                return
            st.session_state.api_key = api_key

        # Initialize the agent
        agent, error = initialize_agent(df, api_key)

        if error:
            st.error(f"Error initializing agent: {error}")
            return

        # Store the agent and data in session state
        st.session_state.agent = agent
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name

        # Success message
        st.success(f"File '{uploaded_file.name}' uploaded and processed successfully!")

        # Add a welcome message to chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        if not st.session_state.chat_history:
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "type": "text",
                    "content": f"I've analyzed your data from '{uploaded_file.name}'. You can now ask me questions about it!",
                }
            )

        # Add button to continue to chat
        if st.button(
            "Continue to Chat", type="primary", use_container_width=True, key="continue_to_chat_btn"
        ):
            st.rerun()
    else:
        st.info("Please upload a CSV or Excel file to get started.")
