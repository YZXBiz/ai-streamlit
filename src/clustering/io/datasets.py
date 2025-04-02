"""Read/Write datasets from/to external sources/destinations."""

# %% IMPORTS
import abc
import io
import json
import os
import pickle
import typing as T
import warnings
from io import BytesIO
from pathlib import Path

import duckdb
import polars as pl
import pydantic as pdt
import snowflake.connector
from azure.storage.blob import BlobClient
from snowflake.connector.pandas_tools import write_pandas

from clustering.config import SETTINGS

# snowflake has a warning about the sqlalchemy; not important
warnings.filterwarnings("ignore", category=UserWarning)


# %% READERS
class Reader(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset reader.

    Use a reader to load a dataset in memory.
    e.g., to read file, database, cloud storage, ...

    Parameters:
        limit (int, optional): maximum number of rows to read. Defaults to None.
    """

    KIND: str

    limit: int | None = None

    path: str | None = None

    @abc.abstractmethod
    def read(self) -> pl.DataFrame:
        """Read a dataframe from a dataset.

        Returns:
            pl.DataFrame: dataframe representation.
        """


class ParquetReader(Reader):
    """Read a dataframe from a parquet file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["ParquetReader"] = "ParquetReader"

    path: str
    backend: T.Literal["pyarrow", "numpy_nullable"] = "pyarrow"

    @T.override
    def read(self) -> pl.DataFrame:
        data = pl.read_parquet(self.path, dtype_backend="pyarrow")
        if self.limit is not None:
            data = data.head(self.limit)
        return data


class ExcelReader(Reader):
    """Read a dataframe from an Excel file.

    Parameters:
        path (str): local path to the dataset.
        sheet_name (str | int | None): name or index of the sheet to read.
    """

    KIND: T.Literal["ExcelReader"] = "ExcelReader"

    path: str
    sheet_name: str | int | None = 0

    def read(self) -> pl.DataFrame:
        data = pl.read_excel(self.path, sheet_name=self.sheet_name, engine="openpyxl")
        if self.limit is not None:
            data = data.head(self.limit)  # type: ignore
        return data  # type: ignore


class CSVReader(Reader):
    """Read a dataframe from a CSV file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["CSVReader"] = "CSVReader"

    path: str

    def read(self) -> pl.DataFrame:
        data = pl.read_csv(self.path)
        if self.limit is not None:
            data = data.head(self.limit)
        return data


class PickleReader(Reader):
    """Read a dataframe from a Pickle file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["PickleReader"] = "PickleReader"

    path: str

    def read(self) -> pl.DataFrame:
        with open(self.path, "rb") as file:
            data = pickle.load(file)
        if self.limit is not None:
            data = data.head(self.limit)
        return data


class SnowflakeReader(Reader):
    """Read a dataframe from a Snowflake database.
    Reading a Big Data is resource intensive and cache is implemented using DuckDB.db.

    Parameters:
        query (str): SQL query to execute.
    """

    KIND: T.Literal["SnowflakeReader"] = "SnowflakeReader"

    # Must Provide
    query: str

    # Optional
    use_cache: bool = True
    cache_file: str = str(Path(SETTINGS.ROOT_DIR) / "cache/snowflake_cache.duckdb")

    pkb_path: str = str(Path(SETTINGS.ROOT_DIR) / "creds/pkb.pkl")
    creds_path: str = str(Path(SETTINGS.ROOT_DIR) / "creds/sf_creds.json")

    def _create_connection(self) -> snowflake.connector.SnowflakeConnection:
        pkb = pl.read_pickle(self.pkb_path)
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

    def _load_cache(self) -> pl.DataFrame | None:
        if os.path.exists(self.cache_file):
            conn = duckdb.connect(self.cache_file)
            try:
                result = conn.execute("SELECT data FROM cache WHERE query = ?", (self.query,)).fetchone()
                if result:
                    # Deserialize the binary format back to a DataFrame
                    data = pl.read_parquet(io.BytesIO(result[0]))
                    return data
            except duckdb.CatalogException as e:
                print("DuckDB CatalogException:", e)
        return None

    def _save_cache(self, data: pl.DataFrame) -> None:
        conn = duckdb.connect(self.cache_file)
        conn.execute("CREATE TABLE IF NOT EXISTS cache (query VARCHAR, data BLOB)")
        data_blob = data.to_parquet()
        conn.execute("INSERT INTO cache VALUES (?, ?)", (self.query, data_blob))

    def read(self) -> pl.DataFrame:
        # Check if the query result is already cached
        if self.use_cache:
            cached_data = self._load_cache()
            if cached_data is not None:
                return cached_data

        conn = self._create_connection()
        # * polars is faster than pandas
        df = pl.read_database(
            query=self.query,
            connection=conn,
        )
        # Cache the result
        self._save_cache(df)

        if self.limit is not None:
            df = df.head(self.limit)
        return df


class BlobReader(Reader):
    """Read a dataframe from a blob storage.

    Parameters:
        blob_name (str): Name of the blob.
            It must include the following file extensions: .csv, .parquet, .pkl, or .pickle.
    """

    KIND: T.Literal["BlobReader"] = "BlobReader"

    blob_name: str

    # Configuration
    max_concurrency: int = 8

    def read(self) -> pl.DataFrame:
        _blob_client = BlobClient(
            account_url=SETTINGS.ACCOUNT_URL,
            container_name=SETTINGS.CONTAINER_NAME,
            blob_name=self.blob_name,
            credential=SETTINGS.AZURE_CREDS,
        )

        try:
            # Download the blob content with parallelism
            blob_data = _blob_client.download_blob(max_concurrency=self.max_concurrency).readall()
        except Exception as e:
            raise RuntimeError(f"Failed to download blob: {e}")

        # Determine the file format from the blob name extension
        file_extension = os.path.splitext(self.blob_name)[1].lower()

        if file_extension == ".csv":
            data = pl.read_csv(BytesIO(blob_data))
        elif file_extension == ".parquet":
            data = pl.read_parquet(BytesIO(blob_data))
        elif file_extension == ".pkl" or file_extension == ".pickle":
            data = pickle.loads(blob_data)
        else:
            raise ValueError("Unsupported file format. Please use a CSV, Parquet, or Pickle file.")

        return data


ReaderKind = T.Union[
    ParquetReader,
    ExcelReader,
    CSVReader,
    PickleReader,
    SnowflakeReader,
    BlobReader,
]


# %% WRITERS
class Writer(abc.ABC, pdt.BaseModel, strict=True, frozen=True, extra="forbid"):
    """Base class for a dataset writer.

    Use a writer to save a dataset from memory.
    e.g., to write file, database, cloud storage, ...
    """

    KIND: str

    @abc.abstractmethod
    def write(self, data: pl.DataFrame | T.Any) -> None:
        """Write a dataframe to a dataset.

        Args:
            data (pl.DataFrame): dataframe representation.
        """


class ParquetWriter(Writer):
    """Writer a dataframe to a parquet file.

    Parameters:
        path (str): local or S3 path to the dataset.
    """

    KIND: T.Literal["ParquetWriter"] = "ParquetWriter"

    path: str

    def write(self, data: pl.DataFrame) -> None:
        pl.DataFrame.to_parquet(data, self.path)


class ExcelWriter(Writer):
    """Write a dataframe to an Excel file.

    Parameters:
        path (str): local path to the dataset.
        sheet_name (str): name of the sheet to write.
    """

    KIND: T.Literal["ExcelWriter"] = "ExcelWriter"

    path: str
    sheet_name: str = "Store-Cluster"

    def write(self, data: pl.DataFrame) -> None:
        pl.DataFrame.to_excel(data, self.path, sheet_name=self.sheet_name, index=False)


class CSVWriter(Writer):
    """Write a dataframe to a CSV file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["CSVWriter"] = "CSVWriter"

    path: str

    def write(self, data: pl.DataFrame) -> None:
        pl.DataFrame.to_csv(self=data, path_or_buf=self.path, index=False)


class PickleWriter(Writer):
    """Write a dataframe to a Pickle file.

    Parameters:
        path (str): local path to the dataset.
    """

    KIND: T.Literal["PickleWriter"] = "PickleWriter"

    path: str

    def write(self, data: pl.DataFrame) -> None:
        with open(self.path, "wb") as file:
            pickle.dump(data, file)


class SnowflakeWriter(Writer):
    """Write a dataframe to a Snowflake database.

    Parameters:
        table (str): name of the table to write.
    """

    KIND: T.Literal["SnowflakeWriter"] = "SnowflakeWriter"

    # Must Provide
    table: str

    pkb_path: str = str(Path(SETTINGS.ROOT_DIR) / "creds/pkb.pkl")
    creds_path: str = str(Path(SETTINGS.ROOT_DIR) / "creds/sf_creds.json")

    def _create_connection(self) -> snowflake.connector.SnowflakeConnection:
        pkb = pl.read_pickle(self.pkb_path)
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

    def write(self, data: pl.DataFrame) -> None:
        conn = self._create_connection()
        write_pandas(
            conn=conn,  # type: ignore
            df=data,
            table_name=self.table,
            database="DL_FSCA_SLFSRV",
            auto_create_table=True,
            overwrite=True,
            schema="TWA07",
        )


class BlobWriter(Writer):
    """Write a pandas dataframe or other Python object to blob storage.

    By default, the file will be saved in the root of the container.

    Parameters:
        blob_name (str): Name of the blob in the container. Please add the file extension to the name and match the file type.
                         e.g., "data.parquet" or "data.csv".
        overwrite (bool): Whether to overwrite the blob if it already exists. Defaults to True.
    """

    KIND: T.Literal["BlobWriter"] = "BlobWriter"

    blob_name: str
    overwrite: bool = True

    # Configuration
    max_concurrency: int = 8

    def write(self, data) -> None:
        buffer = BytesIO()
        file_extension = os.path.splitext(self.blob_name)[1].lower()

        if isinstance(data, pl.DataFrame):
            if file_extension == ".csv":
                data.to_csv(buffer, index=False)
            elif file_extension == ".parquet":
                data.to_parquet(buffer, index=False)
            else:
                raise ValueError("Unsupported file format for dataframe. Please use a CSV or Parquet file.")
        else:
            if file_extension in [".pkl", ".pickle"]:
                pickle.dump(data, buffer)
            else:
                raise ValueError("Unsupported file format for non-dataframe. Please use a Pickle file.")

        buffer.seek(0)

        blob_client = BlobClient(
            account_url=SETTINGS.ACCOUNT_URL,
            container_name=SETTINGS.CONTAINER_NAME,
            blob_name=self.blob_name,
            credential=SETTINGS.AZURE_CREDS,
        )

        try:
            blob_client.upload_blob(
                buffer,
                blob_type="BlockBlob",
                overwrite=self.overwrite,
                max_concurrency=self.max_concurrency,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upload blob: {e}")


WriterKind = T.Union[
    ParquetWriter,
    ExcelWriter,
    CSVWriter,
    PickleWriter,
    SnowflakeWriter,
    BlobWriter,
]
