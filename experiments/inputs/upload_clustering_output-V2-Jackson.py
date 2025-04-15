#!/usr/bin/env python

# # Dependencies

# %pip install fsutils --user
import difflib
import os

# from utils.utils import worker_output_table_validation
import pandas as pd
from fsutils import config
from fsutils import run_sf_sql as rp

# from pandasql import sqldf
from snowflake.connector.pandas_tools import write_pandas

# Pandas Display Settings
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)  # Set width to None to avoid wrapping


# # Utils


def get_last_datetime_token(filename: str) -> str:
    """
    Extract the date/time token at the end of filenames like:
    merged_clusters_PERSONAL_CLEANSING_20250212_1301.csv
    Returning '20250212_1301'.
    """
    if not filename.endswith(".csv"):
        return None  # skip anything that isn't a CSV
    no_ext = filename.split(".")[0]  # e.g., merged_clusters_PERSONAL_CLEANSING_20250212_1301
    tokens = no_ext.split("_")
    if len(tokens) < 3:
        return None  # skip anything that doesn't have at least two underscores
    return tokens[-2] + "_" + tokens[-1]  # e.g. 20250212_1301


def get_last_datetime_token(filename: str) -> str:
    """
    Extract the date/time token at the end of filenames like:
    merged_clusters_PERSONAL_CLEANSING_20250212_1301.csv
    Returning '20250212_1301'.
    """
    if not filename.endswith(".csv"):
        return None  # skip anything that isn't a CSV
    no_ext = filename.split(".")[0]
    tokens = no_ext.split("_")
    if len(tokens) < 3:
        return None
    return tokens[-2] + "_" + tokens[-1]


def normalize_str(s: str) -> str:
    """
    A helper to normalize the strings for matching.
    Example: 'HAND_&_BODY' -> 'HAND & BODY', 'ORAL_HYGIENE' -> 'ORAL HYGIENE', etc.
    Adjust as needed for your use-case.
    """
    # Replace `_&_` with ` & `
    s = s.replace("_&_", " & ")
    # Replace all remaining underscores with spaces
    s = s.replace("_", " ")
    # Strip and make uppercase (or whichever case you prefer)
    s = s.strip().upper()
    return s


def get_closest_match(query: str, valid_list: list) -> str:
    """
    Use difflib to get the closest match from valid_list to the query string.
    Returns None if there is no close match at all, else returns the best guess.
    """
    # difflib.get_close_matches(query, possibilities, n=1, cutoff=0.0)
    # will return a list of the single best match, or empty if none.
    matches = difflib.get_close_matches(query, valid_list, n=1, cutoff=0.0)
    return matches[0] if matches else None


# Finally, create a new column in df_filtered called "CAT_DSC"
def map_category_to_cat_dsc(category_value: str) -> str:
    # Step 1: Normalize the category_value
    norm_cat = normalize_str(category_value)
    # Step 2: Find the closest normalized official
    best_normed = get_closest_match(norm_cat, normalized_official)
    if best_normed is not None:
        # Return the original official category text
        return normalized_to_original[best_normed]
    else:
        # Fallback if no match at all
        return category_value  # or some sentinel like "UNKNOWN"


# # Main

directory_path = "/home/jovyan/older code/notebooks/clustering_output/"


def get_last_datetime_token(filename: str) -> str:
    """
    Extract the date/time token at the end of filenames like:
    merged_clusters_PERSONAL_CLEANSING_20250212_1301.csv
    Returning '20250212_1301'.
    """
    if not filename.endswith(".csv"):
        return None  # skip anything that isn't a CSV
    no_ext = filename.split(".")[0]  # e.g., merged_clusters_PERSONAL_CLEANSING_20250212_1301
    tokens = no_ext.split("_")
    if len(tokens) < 3:
        return None  # skip anything that doesn't have at least two underscores
    return tokens[-2] + "_" + tokens[-1]  # e.g. 20250212_1301


all_files = os.listdir(directory_path)
# Filter down to only valid date/time tokens
valid_times = []
for f in all_files:
    dt = get_last_datetime_token(f)
    if dt is not None:
        valid_times.append(dt)
if not valid_times:
    raise ValueError("No valid CSV files found in directory.")


# Find the earliest (smallest) date/time across all files
earliest_datetime = min(valid_times)
print("earliest_datetime =", earliest_datetime)


# Rewrite all files to have the same end path as the earliest date/time
for file in all_files:
    dt = get_last_datetime_token(file)
    if dt is not None:
        new_filename = file.replace(dt, earliest_datetime)
        os.rename(os.path.join(directory_path, file), os.path.join(directory_path, new_filename))


# Now only read files that contain that earliest date/time
dfs = []
for file in os.listdir(directory_path):
    dt = get_last_datetime_token(file)
    if dt == earliest_datetime:
        df = pd.read_csv(os.path.join(directory_path, file))
        dfs.append(df)

# Concatenate the dataframes
dfs = pd.concat(dfs, ignore_index=True)


# Add a category column
dfs["category"] = dfs["external_granularity"].str.split("_").str[:-2].str.join("_")

print("Duplicates :", dfs["store_nbr"].nunique() * dfs["category"].nunique() < dfs.shape[0])
print(
    "1:1 mismatch :",
    dfs["store_nbr"].nunique() * dfs["category"].nunique()
    > dfs.drop_duplicates(subset=["store_nbr", "category"]).shape[0],
)

df_filtered = dfs[
    [
        "store_nbr",
        "category",
        "external_cluster_labels",
        "internal_cluster_labels",
        "demand_cluster_labels",
        "rebalanced_demand_cluster_labels",
    ]
].copy()


df_filtered.category.unique()


df_filtered.nunique()


# ----------------------------------------------------------------------------
# 2) Manually define a dictionary that maps your existing category values
# to the new CAT_DSC strings you'd like.
# ----------------------------------------------------------------------------
category_mapping = {
    # Wave 1
    "VITAMINS": "VITAMINS",
    "ORAL_HYGIENE": "ORAL HYGIENE",
    "BEVERAGES": "BEVERAGES",
    "PERSONAL_CLEANSING": "PERSONAL CLEANSING",
    "HOUSEHOLD": "HOUSEHOLD",
    # Wave 2
    "HOME_DIAGNOSTICS": "HOME DIAGNOSTICS",
    "BABY_CARE": "BABY CARE",
    "ALLERGY_REMEDIES": "ALLERGY REMEDIES",
    "HAND_&_BODY": "HAND & BODY",
    "SNACKS": "SNACKS",
    "STATIONERY": "STATIONERY",
    "SHAVING_NEEDS": "SHAVING NEEDS",
    "FACIAL_CARE": "FACIAL CARE",
    "CHILDRENS_REMEDIES": "CHILDRENS REMEDIES",
    "ADULT_CARE": "ADULT CARE",
    "DIET_NUTRITION": "DIET/NUTRITION",
    "ACNE_HSC": "ACNE/HSC",
    "GROCERY": "GROCERY",
    "COLD_REMEDIES": "COLD REMEDIES",
    "DIGESTIVE_HEALTH": "DIGESTIVE HEALTH",
    "HAIR_CARE": "HAIR CARE",
    "EXTERNAL_PAIN": "EXTERNAL PAIN",
    "FIRST_AID": "FIRST AID",
    "CANDY": "CANDY",
    "DEODORANTS": "DEODORANTS",
    "TRIAL_TRAVEL": "TRIAL TRAVEL",
    "FEMININE_CARE": "FEMININE CARE",
    "HAIR_COLOR": "HAIR COLOR",
    "HOUSEHOLD_PAPER": "HOUSEHOLD PAPER",
    "TEXTURED_HAIR": "TEXTURED HAIR",
}

# ----------------------------------------------------------------------------
# 3) Apply the dictionary to create the CAT_DSC column
# ----------------------------------------------------------------------------
# If a given 'category' value is not in the dictionary, we can either default
# to the original string or set it to something like 'UNKNOWN'
df_filtered["CAT_DSC"] = df_filtered["category"].map(category_mapping).fillna(pd.NA)


missing_categories = df_filtered.loc[df_filtered["CAT_DSC"].isna(), "category"].unique()
assert len(missing_categories) == 0, f"Missing CAT_DSC mapping for categories: {missing_categories}"


# # Normalize the official CAT_DSC strings too, so we compare on the same basis.
# # We'll keep an extra structure mapping normalized -> original official.
# normalized_to_original = {}
# for cat in official_categories:
#     norm = normalize_str(cat)
#     normalized_to_original[norm] = cat  # store the *exact* original

# normalized_official = list(normalized_to_original.keys())

# df_filtered["CAT_DSC"] = df_filtered["category"].apply(map_category_to_cat_dsc)

# ###############################################################################
# # End result
# ###############################################################################
# print(df_filtered.head(10))


df_filtered["CAT_DSC"].value_counts()


df_filtered["category"].value_counts()


df_filtered["CAT_DSC"].unique()
# array(['BEVERAGES', 'VITAMINS', 'PERSONAL_CLEANSING', 'HOUSEHOLD',
#        'ORAL_HYGIENE'], dtype=object)


df_filtered.shape
# (34145, 6)


df_filtered.head()


df_filtered.CAT_DSC.unique()


# # Export

latest_datetime = earliest_datetime


# -------------------------------------------------------------------
# Snowflake-related imports and writing to Snowflake
# -------------------------------------------------------------------

# Define the config map name
configmapname = "notebook-medium"

# Get configs
params = config.get_config(configmapname)

# Get the SF connection
conn, cur = rp.get_connection(configmapname)

df = df_filtered.copy()

# 7) Form our table name using the latest date/time
#    (if you only want the date, you could parse out the YYYYMMDD)
# table_name = f"ASSORTMENT_STORE_CLUSTERS_{latest_datetime}"
table_name = "FINAL_ASSORTMENT_STORE_CLUSTERS"

# 8) Write to Snowflake
write_pandas(
    conn,
    df,
    table_name,
    database="DL_FSCA_SLFSRV",
    schema="TWA07",
    overwrite=True,
    auto_create_table=True,
)

print(f"Data written to table: {table_name}")


# ### Appending

master_name = "FINAL_ASSORTMENT_STORE_CLUSTERS_ARCHIVE"


df_archive = df.copy()
# add a column called timestamp and use this variable latest_datetime
df_archive["timestamp"] = latest_datetime


write_pandas(
    conn,
    df_archive,
    master_name,
    database="DL_FSCA_SLFSRV",
    schema="TWA07",
    overwrite=False,
    auto_create_table=True,
)


# ## Adhoc

dighealth = pd.read_sql(
    "select * from DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250321_1842_DIGHEALTH", conn
)
snacks = pd.read_sql(
    "select * from DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250401_1553_SNACKS", conn
)
candy = pd.read_sql(
    "select * from DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250402_1949_CANDY", conn
)
final_assortment = pd.read_sql(
    "select * from DL_FSCA_SLFSRV.TWA07.FINAL_ASSORTMENT_STORE_CLUSTERS", conn
)


dighealth.head(2)


snacks = snacks.rename(
    columns={"STORE_NBR": "store_nbr", "CLUSTER": "rebalanced_demand_cluster_labels"}
)


candy.head(2)


final_assortment.head(2)


# Merge all DataFrames
merged_df = pd.concat([dighealth, snacks, candy, final_assortment], ignore_index=True)


merged_df


write_pandas(
    conn,
    merged_df,
    f"FINAL_ASSORTMENT_STORE_CLUSTERS_{latest_datetime}_MERGED",
    database="DL_FSCA_SLFSRV",
    schema="TWA07",
    overwrite=True,
    auto_create_table=True,
)


f"FINAL_ASSORTMENT_STORE_CLUSTERS_{latest_datetime}_MERGED"


import pandas as pd
from fsutils import config
from fsutils import run_sf_sql as rp
from snowflake.connector.pandas_tools import write_pandas

conn, cur = rp.get_connection("notebook-medium")
import pandas as pd

# Define the table names
tables = [
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250312_1859",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250313_1956",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250319_1339",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250319_1934",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250320_1843",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250321_1446",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250321_1842",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250324_2027",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250325_1347",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250325_1719",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250326_1342",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250326_1529",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250326_1717",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250326_1901",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250327_1912",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250328_1901",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250331_1351",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250331_1701",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250401_1553",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250402_1949",
    "DL_FSCA_SLFSRV.TWA07.ASSORTMENT_STORE_CLUSTERS_20250403_1746",
]

# Additional table with its own timestamp
table2 = ["DL_FSCA_SLFSRV.TWA07.FINAL_ASSORTMENT_STORE_CLUSTERS_ARCHIVE"]


# Define a function to preprocess timestamps
def preprocess_timestamp(timestamp):
    try:
        # If the timestamp is already in a valid format, return it
        return pd.to_datetime(timestamp)
    except:
        # Handle custom format like '20250404_1615'
        if "_" in timestamp:
            date_part, time_part = timestamp.split("_")
            formatted_timestamp = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]} {time_part[:2]}:{time_part[2:]}:00"
            return pd.to_datetime(formatted_timestamp)
        else:
            raise ValueError(f"Unknown timestamp format: {timestamp}")


# Load data from each table, add timestamp column, and concatenate
dataframes = []

# Process tables with timestamps
for table in tables:
    # Extract the date from the table name
    date_str = table.split("_")[-2]  # Extract '20250312' part
    timestamp = pd.to_datetime(date_str, format="%Y%m%d")  # Convert to datetime

    # Query the table
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, conn)  # Replace 'conn' with your database connection object

    # Add the timestamp column
    df["timestamp"] = timestamp

    # Append to the list of DataFrames
    dataframes.append(df)

# Concatenate all DataFrames from the first set of tables
final_df = pd.concat(dataframes, ignore_index=True)

# Process the additional table (table2) with its own timestamp
for table in table2:
    query = f"SELECT * FROM {table}"
    df_table2 = pd.read_sql(query, conn)  # Replace 'conn' with your database connection object

    # Ensure the timestamp column exists in table2
    if "timestamp" not in df_table2.columns:
        raise ValueError(f"The table {table} does not have a 'timestamp' column.")

    # Append table2 to the final DataFrame
    final_df = pd.concat([final_df, df_table2], ignore_index=True)

# Enforce consistent timestamp format
final_df["timestamp"] = final_df["timestamp"].apply(preprocess_timestamp)


write_pandas(
    conn,
    final_df,
    "FINAL_ASSORTMENT_STORE_CLUSTERS_MERGED_MICHELLE_Apr_10_2025",
    database="DL_FSCA_SLFSRV",
    schema="TWA07",
    overwrite=True,
    auto_create_table=True,
)
