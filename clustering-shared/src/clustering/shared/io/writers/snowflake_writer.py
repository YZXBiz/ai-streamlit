"""Snowflake writer implementation."""

import json
import pickle

import polars as pl
import snowflake.connector

from shared.io.writers.base import Writer


class SnowflakeWriter(Writer):
    """Writer for Snowflake database."""

    # Required parameters
    table: str
    database: str = "DL_FSCA_SLFSRV"
    schema: str = "TWA07"

    # Options
    auto_create_table: bool = True
    overwrite: bool = True

    # Credentials paths
    pkb_path: str = "creds/pkb.pkl"
    creds_path: str = "creds/sf_creds.json"

    def _create_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Create a Snowflake connection.

        Returns:
            SnowflakeConnection: A connection to Snowflake
        """
        with open(self.pkb_path, "rb") as file:
            pkb = pickle.load(file)

        with open(self.creds_path) as file:
            sf_params = json.loads(file.read())

        conn = snowflake.connector.connect(
            user=sf_params["SF_USER_NAME"],
            private_key=pkb,
            account=sf_params["SF_ACCOUNT"],
            database=sf_params["SF_DB"],
            warehouse=sf_params["SF_WAREHOUSE"],
            role=sf_params["SF_USER_ROLE"],
            insecure_mode=sf_params.get("SF_INSECURE_MODE") == "True",
        )
        return conn

    def _write_to_destination(self, data: pl.DataFrame) -> None:
        """Write data to Snowflake.

        Args:
            data: DataFrame to write
        """
        conn = self._create_connection()

        # Convert to pandas first since Snowflake connector expects pandas
        pandas_df = data.to_pandas()

        # Use Snowflake's pandas_tools to write the DataFrame
        from snowflake.connector.pandas_tools import write_pandas

        write_pandas(
            conn=conn,
            df=pandas_df,
            table_name=self.table,
            database=self.database,
            schema=self.schema,
            auto_create_table=self.auto_create_table,
            overwrite=self.overwrite,
        )

        conn.close()
