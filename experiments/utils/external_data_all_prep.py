import os
import logging
import pandas as pd

# Import the internal pipeline configuration
from configs.internal_config import DATA_DIR

# Import all needed column definitions from config
# (Make sure you've added DROP_COLS, SPEND_COLS, etc. in your configs.config)
from configs.config import (
    DROP_COLS,
    SPEND_COLS,
    COTENANT_COLS,
    COMP_COLS,
    PLACER_COLS,
    COLUMNS_TO_DROP,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_and_clean_data():
    """
    Loads and merges multiple CSV files containing store data,
    cleans them (e.g., fills missing data, drops certain columns),
    and returns a final merged DataFrame.

    Steps:
        1. Load the 'df_additional' DataFrame and drop specified columns.
        2. Load and merge additional data: spend, cotenant, competitor, and Placer.ai data.
        3. Drop rows with any missing values.
        4. Drop additional columns (COLUMNS_TO_DROP) before returning.
    """
    logger.info("Starting data load and cleaning process from DATA_DIR: %s", DATA_DIR)

    # ------------------------------
    # 1. Load the base DataFrame (Originally df_start, now replaced by df_additional)
    #    and drop unused columns
    # ------------------------------
    logger.info("Loading base data (df_additional).")
    additional_path = os.path.join(DATA_DIR, "clustering_features_plus_night_traffic_07022025.csv")
    df_additional = pd.read_csv(additional_path)

    logger.info("Dropping columns: %s", DROP_COLS)
    df_base_clean = df_additional.drop(columns=DROP_COLS, errors="ignore")
    logger.debug("df_base_clean shape: %s", df_base_clean.shape)

    # ------------------------------
    # 2. Load & Clean Additional Data (spend, cotenant, competition, Placer.ai)
    # ------------------------------

    # 2a. Spend Data
    logger.info("Loading Spend Data.")
    spent_path = os.path.join(
        DATA_DIR,
        "Urbanicity CVS Drive Times with Demos and competitors_#COMP_DEMO_ALL_07022025.csv",
    )
    spent_df = pd.read_csv(spent_path)
    spent_df_clean = spent_df[SPEND_COLS].copy().fillna(0)
    logger.debug("spent_df_clean shape: %s", spent_df_clean.shape)

    # 2b. Cotenants
    logger.info("Loading Cotenants Data.")
    cotenant_path = os.path.join(DATA_DIR, "Cotenants_RSP_state_07022025.csv")
    cotenant_df = pd.read_csv(cotenant_path)
    cotenant_df_clean = cotenant_df[COTENANT_COLS].fillna(0)
    logger.debug("cotenant_df_clean shape: %s", cotenant_df_clean.shape)

    # 2c. Competition
    logger.info("Loading Competition Data.")
    competition_path = os.path.join(
        DATA_DIR,
        "Urbanicity CVS Drive Times with Demos and competitors_#COMP_DEMO_ALL_07022025.csv",
    )
    competition_df = pd.read_csv(competition_path)
    competition_df_clean = competition_df[COMP_COLS].fillna(0)
    logger.debug("competition_df_clean shape: %s", competition_df_clean.shape)

    # 2d. Placer.ai Data
    logger.info("Loading Placer.ai Data.")
    placer_path = os.path.join(
        DATA_DIR, "placer_store_features_2024_20250211_1420_with_remaining.csv"
    )
    df_placer = pd.read_csv(placer_path)
    df_placer_clean = df_placer[PLACER_COLS]
    # Drop rows without 'unique_visitors' to ensure full data coverage
    df_placer_clean = df_placer_clean[df_placer_clean["unique_visitors"].notna()]
    logger.debug("df_placer_clean shape (after dropping nulls): %s", df_placer_clean.shape)

    # 2e.
    # TODO: add here temporarily
    #     from fsutils import run_sf_sql as rp
    #     import pandas as pd

    #     # Get the Snowflake connection and cursor
    #     conn, _ = rp.get_connection("notebook-xlarge")
    #     query = "select STORE_NBR, hr24_ind from CORE_FSSC.CURATED_LOCATION.STORE;"
    #     df_24hr = pd.read_sql(query, conn)

    # ------------------------------
    # 3. Perform incremental merges
    # ------------------------------
    logger.info("Merging all data sources.")

    # a. Merge base with spend
    df_m1 = df_base_clean.merge(spent_df_clean, how="left", on="STORE_NBR")

    # b. Merge with cotenant
    df_m2 = df_m1.merge(cotenant_df_clean, how="left", on="STORE_NBR")
    df_m2["tenant_cnt"] = df_m2["tenant_cnt"].fillna(0)

    # c. Merge with competition
    df_m3 = df_m2.merge(competition_df_clean, how="left", on="STORE_NBR")

    # d. Merge with Placer.ai data (inner join)
    df_m4 = df_m3.merge(df_placer_clean, how="inner", on="STORE_NBR")
    logger.debug("df_m4 shape after merges: %s", df_m4.shape)

    # e. Merge with 24 hr indicator
    # df_m5 = df_m4.merge(df_24hr, how='inner', on='STORE_NBR')
    # logger.debug("df_m5 shape after merges: %s", df_m5.shape)

    # ------------------------------
    # 4. Identify & remove rows with missing data
    # ------------------------------
    logger.info("Identifying rows with missing data.")
    df_final = df_m4.copy()
    missing_rows = df_final[df_final.isnull().any(axis=1)]
    logger.info("Number of rows with missing data: %d", missing_rows.shape[0])

    for idx, row in missing_rows.iterrows():
        store_nbr = row["STORE_NBR"]
        proportion_missing = row.isnull().mean()
        logger.debug("STORE_NBR: %s, proportion missing: %.2f", store_nbr, proportion_missing)

    logger.info("Dropping all rows with any missing values.")
    df_final.dropna(inplace=True)
    logger.debug("df_final shape after dropping missing rows: %s", df_final.shape)

    # Drop columns from the config's COLUMNS_TO_DROP
    logger.info("Dropping columns specified in config: %s", COLUMNS_TO_DROP)
    df_final.drop(columns=COLUMNS_TO_DROP, errors="ignore", inplace=True)
    logger.debug("df_final shape after final column drops: %s", df_final.shape)

    logger.info("Data cleanup complete. Returning the final DataFrame.")
    return df_final


def load_clustered_csvs(directory_path):
    """
    Loads all CSV files from 'directory_path' whose names follow the pattern:
        df_clustered_<KEY>_<YYYYMMDD>_<HHMM>.csv
    and returns a dictionary mapping <KEY> -> DataFrame.
    """
    data_dict = {}
    for filename in os.listdir(directory_path):
        # Check for the "df_clustered_" prefix and ".csv" suffix
        if filename.startswith("df_clustered_") and filename.endswith(".csv"):
            # Remove the prefix ("df_clustered_") and the suffix (".csv")
            stripped_name = filename[len("df_clustered_") : -4]

            # Split from the right on underscores, twice
            # e.g. "GROCERY_20250123_2051" -> ["GROCERY", "20250123", "2051"]
            parts = stripped_name.rsplit("_", 2)

            # The key is everything before the date/time parts
            key = parts[0]

            file_path = os.path.join(directory_path, filename)
            df = pd.read_csv(file_path)

            # Store in the dictionary
            data_dict[key] = df

    return data_dict
