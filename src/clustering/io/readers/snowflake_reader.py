"""Snowflake reader implementation."""

import io
import json
import os
import pickle
from typing import Optional

import duckdb
import polars as pl
import snowflake.connector

from clustering.io.readers.base import Reader


class SnowflakeReader(Reader):
    """Reader for Snowflake database.

    Reading data from Snowflake can be resource intensive, so caching is implemented using DuckDB.
    """

    query: str
    use_cache: bool = True
    cache_file: str = "cache/snowflake_cache.duckdb"

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

        with open(self.creds_path, "r") as file:
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

    def _load_cache(self) -> Optional[pl.DataFrame]:
        """Load cached query results if available.

        Returns:
            Optional[pl.DataFrame]: Cached dataframe or None if not cached
        """
        if os.path.exists(self.cache_file):
            conn = duckdb.connect(self.cache_file)
            try:
                result = conn.execute("SELECT data FROM cache WHERE query = ?", (self.query,)).fetchone()
                if result:
                    return pl.read_parquet(io.BytesIO(result[0]))
            except duckdb.CatalogException as e:
                print("DuckDB CatalogException:", e)
        return None

    def _save_cache(self, data: pl.DataFrame) -> None:
        """Save query results to cache.

        Args:
            data: DataFrame to cache
        """
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)

        conn = duckdb.connect(self.cache_file)
        conn.execute("CREATE TABLE IF NOT EXISTS cache (query VARCHAR, data BLOB)")

        # Convert DataFrame to parquet bytes
        buffer = io.BytesIO()
        data.write_parquet(buffer)
        buffer.seek(0)

        conn.execute("INSERT INTO cache VALUES (?, ?)", (self.query, buffer.getvalue()))

    def read(self) -> pl.DataFrame:
        """Read data from Snowflake.

        Returns:
            DataFrame containing the data
        """
        # Check if the query result is already cached
        if self.use_cache:
            cached_data = self._load_cache()
            if cached_data is not None:
                return cached_data

        # Execute query if not cached or cache disabled
        conn = self._create_connection()

        # Use polars to read directly from the database connection
        data = pl.read_database(query=self.query, connection=conn)
        conn.close()

        # Cache the result
        if self.use_cache:
            self._save_cache(data)

        if self.limit is not None:
            data = data.head(self.limit)

        return data
