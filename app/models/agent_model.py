import logging
from typing import Any

import pandas as pd
from pandasai import Agent, DataFrame
from pandasai_openai import OpenAI


class AgentModel:
    """
    Model for managing PandasAI agents and their configuration.
    """

    def __init__(self) -> None:
        self.default_config: dict[str, Any] = {}
        self.default_config: dict[str, Any] = {
            # "save_charts": False,
            # "verbose": True,
            # "return_intermediate_steps": False,
            # "enable_cache": False,  # Disable cache to prevent issues with multiple dataframes
            "temperature": 0.5,
        }

    def create_agent(
        self, data: pd.DataFrame | dict[str, pd.DataFrame], api_key: str
    ) -> tuple[Agent | None, str | None]:
        """
        Create a PandasAI agent with the given dataframe(s).

        Args:
            data: Either a single pandas DataFrame or a dictionary of DataFrames
            api_key: OpenAI API key

        Returns:
            A tuple of (PandasAI agent, error message)
        """
        if not api_key:
            return None, "Missing OpenAI API Key"

        try:
            # Initialize the LLM
            llm = OpenAI(api_token=api_key)

            # Create agent config with LLM
            config = {
                "llm": llm,
                **self.default_config,
            }

            # Handle single dataframe case
            if isinstance(data, pd.DataFrame):
                # Convert pandas DataFrame to PandasAI DataFrame
                pai_df = DataFrame(data)

                # Create agent with the dataframe
                agent = Agent(dfs=pai_df, config=config)
                logging.info(f"Created agent with a single dataframe of shape {data.shape}")

            # Handle multiple dataframes case
            else:
                # Convert each pandas DataFrame to PandasAI DataFrame
                dataframes = []
                dataframe_names = []  # Store table names separately

                # Log information about the dataframes
                logging.info(f"Creating agent with {len(data)} dataframes:")
                for name, df in data.items():
                    # Create PandasAI DataFrame (passing name as metadata, but we won't try to access it directly)
                    smart_df = DataFrame(df, name=name)
                    dataframes.append(smart_df)
                    dataframe_names.append(name)  # Track the name separately
                    logging.info(f"  - {name}: {df.shape}")

                # Create agent with list of dataframes
                agent = Agent(dfs=dataframes, config=config)

                # Train the agent to be aware of multiple tables if there are more than one
                if len(dataframes) > 1:
                    # Use our separately tracked names, not df.name
                    tables_str = ", ".join(f"'{name}'" for name in dataframe_names)
                    agent.add_message(
                        "system",
                        f"You have access to {len(dataframe_names)} tables: {tables_str}. When answering questions, use all relevant tables and mention which tables you're using in your response.",
                    )

            return agent, None

        except Exception as e:
            logging.error(f"Error creating agent: {str(e)}")
            return None, str(e)

    def process_question(
        self, agent: Agent, question: str, is_first_question: bool, output_type: str = None
    ) -> Any:
        """
        Process a user question through the agent.

        Args:
            agent: The PandasAI agent
            question: The user question to process
            is_first_question: Whether this is the first question in the conversation
            output_type: The desired output type ("auto", "string", "dataframe", "chart").
                         Defaults to None (auto)

        Returns:
            The agent's response
        """
        try:
            if is_first_question:
                return agent.chat(question, output_type=output_type)
            else:
                return agent.follow_up(question, output_type=output_type)
        except Exception as e:
            raise Exception(f"Error processing question: {str(e)}") from e
