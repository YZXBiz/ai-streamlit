import logging
import string
from datetime import datetime
from typing import Any

import pandas as pd
from fsutils import run_sf_sql as rp
from snowflake.connector.pandas_tools import write_pandas

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration: alternate names for product_id
ALTERNATE_PRODUCT_ID_COLUMNS = ["item_nbr", "item_no_nbr", "item_id"]


def standardize_to_be_dropped(value: Any) -> str:
    """
    Standardize a value by removing punctuation, converting to lowercase, and trimming whitespace.

    Args:
        value: The input value to standardize

    Returns:
        Standardized string value
    """
    val_str = str(value)
    # Remove punctuation
    val_str_no_punc = val_str.translate(str.maketrans("", "", string.punctuation))
    # Lowercase and strip
    return val_str_no_punc.lower().strip()


def get_snowflake_connection():
    """
    Establish connection to Snowflake.

    Returns:
        Tuple containing connection and cursor objects
    """
    return rp.get_connection("notebook-xlarge")


def process_need_states_data(conn) -> pd.DataFrame:
    """
    Load and process need states data from Snowflake.

    Args:
        conn: Snowflake connection object

    Returns:
        Processed DataFrame containing need states data
    """
    # Read data from Snowflake
    ns_data = pd.read_sql("select * from DL_FSCA_SLFSRV.TWA07.FINAL_NEED_STATES_NEW_ITEMS", conn)

    # Filter out new items
    ns_data = ns_data.query("`New Item` == False")

    # Standardize column names
    ns_data.columns = ns_data.columns.str.upper()

    # Select and deduplicate relevant columns
    ns_data = ns_data[
        [
            "PRODUCT_ID",
            "CATEGORY",
            "NEED_STATE",
            "CDT",
            "ATTRIBUTE_1",
            "ATTRIBUTE_2",
            "ATTRIBUTE_3",
            "ATTRIBUTE_4",
            "ATTRIBUTE_5",
            "ATTRIBUTE_6",
        ]
    ].drop_duplicates()

    # Remove rows with missing need state
    ns_data = ns_data[~ns_data["NEED_STATE"].isna()]

    # Convert attribute columns to strings
    for attribute in [
        "ATTRIBUTE_1",
        "ATTRIBUTE_2",
        "ATTRIBUTE_3",
        "ATTRIBUTE_4",
        "ATTRIBUTE_5",
        "ATTRIBUTE_6",
    ]:
        ns_data[attribute] = ns_data[attribute].astype(str)

    return ns_data


def save_to_csv(data: pd.DataFrame, date_str: str | None = None) -> str:
    """
    Save processed data to a CSV file.

    Args:
        data: DataFrame to save
        date_str: Date string to use in filename (defaults to current date)

    Returns:
        Path to the saved CSV file
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    output_path = (
        f"/home/jovyan/fsassortment/store_clustering/data/need_states_mapping_{date_str}_AM.csv"
    )
    data.to_csv(output_path, index=False)
    logger.info(f"Saved data to {output_path}")

    return output_path


def write_to_snowflake(conn, data: pd.DataFrame, date_str: str | None = None) -> str:
    """
    Write data to Snowflake table.

    Args:
        conn: Snowflake connection
        data: DataFrame to upload
        date_str: Date string to use in table name (defaults to current date)

    Returns:
        Name of the created table
    """
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    table_name = f"NEED_STATES_{date_str}_AM"

    write_pandas(
        conn,
        data,
        table_name,
        database="DL_FSCA_SLFSRV",
        schema="TWA07",
        auto_create_table=True,
        overwrite=True,
    )

    logger.info(f"Wrote data to Snowflake table {table_name}")
    return table_name


def main() -> None:
    """
    Main function to process need states data and upload to Snowflake.
    """
    # Get current date
    current_date = datetime.now().strftime("%Y%m%d")

    # Connect to Snowflake
    conn, _ = get_snowflake_connection()

    # Process data
    ns_data = process_need_states_data(conn)

    # Save to CSV
    save_to_csv(ns_data, current_date)

    # Write to Snowflake
    write_to_snowflake(conn, ns_data, current_date)


if __name__ == "__main__":
    main()
