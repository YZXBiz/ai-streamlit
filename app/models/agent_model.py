from typing import Any

import pandas as pd
from pandasai import Agent, DataFrame
from pandasai_openai import OpenAI


class AgentModel:
    """
    Model for managing PandasAI agents and their configuration.
    """

    def __init__(self) -> None:
        self.default_config: dict[str, bool] = {
            "save_charts": False,
            "verbose": True,
            "return_intermediate_steps": False,
        }

    def create_agent(self, df: pd.DataFrame, api_key: str) -> tuple[Agent | None, str | None]:
        """
        Create a PandasAI agent with the given dataframe.

        Args:
            df: The pandas DataFrame to be analyzed
            api_key: OpenAI API key

        Returns:
            A tuple of (PandasAI agent, error message)
        """
        if not api_key:
            return None, "Missing OpenAI API Key"

        try:
            # Initialize the LLM
            llm = OpenAI(api_token=api_key)

            # Convert pandas DataFrame to PandasAI DataFrame
            pai_df = DataFrame(df)

            # Create agent with the dataframe and LLM config
            agent = Agent(
                dfs=pai_df,
                config={
                    "llm": llm,
                    **self.default_config,
                },
            )

            return agent, None
        except Exception as e:
            return None, str(e)

    def process_question(self, agent: Agent, question: str, is_first_question: bool) -> Any:
        """
        Process a user question through the agent.

        Args:
            agent: The PandasAI agent
            question: The user question to process
            is_first_question: Whether this is the first question in the conversation

        Returns:
            The agent's response
        """
        try:
            if is_first_question:
                return agent.chat(question)
            else:
                return agent.follow_up(question)
        except Exception as e:
            raise Exception(f"Error processing question: {str(e)}") from e
