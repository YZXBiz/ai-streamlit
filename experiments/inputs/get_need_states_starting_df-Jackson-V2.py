# # Dependencies

# %pip install fsutils --user
import pandas as pd
import numpy as np
import time
import logging
import itertools as it
import argparse
import os
import time
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from scipy.spatial import distance
from pandas.api.types import CategoricalDtype
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from fsutils import run_sf_sql as rp, config, email, adls_gen2, log

# from utils.utils import worker_output_table_validation
from multiprocessing import Pool, freeze_support
from snowflake.connector.connection import SnowflakeConnection, SnowflakeCursor
from snowflake.connector.pandas_tools import write_pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# from pandasql import sqldf
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

# Pandas Display Settings
pd.set_option("display.max_rows", 100)  # Show all rows
pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", None)  # Set width to None to avoid wrapping


conn, _ = rp.get_connection("notebook-xlarge")


# Put your excluded store IDs into a Python list
excluded_stores = [
    4615,
    1967,
    2064,
    2280,
    2433,
    2437,
    2724,
    464,
    478,
    509,
    867,
    955,
    1084,
    1113,
    1220,
    1243,
    10289,
    5855,
    5980,
    7400,
    8940,
    8969,
    9106,
    9107,
    3121,
    3226,
    3236,
    3250,
    3296,
    3315,
    3327,
    3351,
    3401,
    3443,
    3570,
    3610,
    3650,
    3655,
    3704,
    3705,
    3706,
    3722,
    3737,
    3756,
    3811,
    3921,
    3956,
    4032,
    4072,
    4701,
    5010,
    5040,
    5041,
    5118,
    5127,
    5136,
    5147,
    5154,
    5233,
    113,
    2379,
    402,
    476,
    536,
    538,
    554,
    564,
    2897,
    591,
    637,
    866,
    925,
    931,
    932,
    976,
    981,
    1126,
    1127,
    1136,
    5783,
    5924,
    6002,
    6732,
    7131,
    7133,
    7897,
    7973,
    8000,
    8378,
    10915,
    3153,
    3986,
    4019,
    4065,
    4789,
    4792,
    5022,
    5499,
    2768,
    2944,
    2963,
    3033,
    3056,
    3061,
    3078,
    10807,
    5775,
    5792,
    5834,
    5841,
    5945,
    6755,
    6772,
    7001,
    7088,
    7127,
    7860,
    8439,
    8778,
    8781,
    8800,
    8803,
    8808,
    8819,
    8824,
    8825,
    8831,
    8840,
    8841,
    8849,
    8854,
    8855,
    8857,
    8858,
    8859,
    8860,
    8865,
    8867,
    8870,
    8874,
    8875,
    8889,
    8892,
    8897,
    8898,
    8943,
    9104,
    9108,
    9115,
    9121,
    9122,
    9123,
    9138,
    9140,
    9151,
    9186,
    9193,
    9196,
    9240,
    9334,
    9349,
    9371,
    9374,
    9377,
    9488,
    9500,
    9501,
    9507,
    9524,
    9557,
    9574,
    9583,
    9588,
    9589,
    9590,
    9594,
    9596,
    9602,
    9604,
    9609,
    9610,
    9617,
    9619,
    9625,
    9626,
    9629,
    9634,
    9636,
    9637,
    9639,
    9642,
    9651,
    9654,
    9664,
    9665,
    9666,
    9672,
    9673,
    9687,
    9690,
    9695,
    9708,
    9728,
    9735,
    9738,
    9745,
    9750,
    9757,
    9762,
    9771,
    9780,
    9782,
    9783,
    9792,
    9794,
    9796,
    9821,
    9837,
    9845,
    9848,
    9850,
    9857,
    9861,
    9866,
    9871,
    9900,
    11073,
    10931,
    1533,
    3178,
    3693,
    4356,
    4706,
    4732,
    4820,
    5260,
    2701,
    213,
    1057,
    1201,
    10412,
    10446,
    10449,
    10485,
    5651,
    5657,
    5857,
    5858,
    5959,
    5970,
    5981,
    6241,
    6665,
    6716,
    6741,
    6742,
    6776,
    6781,
    6786,
    6803,
    6824,
    6847,
    6865,
    6895,
    6979,
    6991,
    7106,
    7149,
    7193,
    7222,
    7231,
    7287,
    7299,
    7306,
    7319,
    7373,
    7403,
    7421,
    7456,
    7664,
    7679,
    7741,
    7748,
    7784,
    7831,
    8389,
    10638,
    8912,
    8962,
    8963,
    10051,
    10073,
    10142,
    10148,
    10215,
    10217,
    10227,
    10946,
]

wave_1_cat = ["HOUSEHOLD", "VITAMINS", "ORAL HYGIENE", "BEVERAGES", "PERSONAL CLEANSING"]

wave_2_cat = [
    "HOME DIAGNOSTICS",
    "BABY CARE",
    "ALLERGY REMEDIES",
    "HAND & BODY",
    "SNACKS",
    "STATIONERY",
]

wave_other = [
    # "SNACKS",
    # "GROCERY",
    # "HOME DIAGNOSTICS",
    # "BABY CARE",
    # "FACIAL CARE",
    # "ALLERGY REMEDIES",
    # "HAND & BODY",
    # "STATIONERY",
    # "HOUSEHOLD",
    # "VITAMINS",
    # "ORAL HYGIENE",
    # "BEVERAGES",
    # "PERSONAL CLEANSING",
    # "CHILDRENS REMEDIES",
    # "ADULT CARE",
    # "SHAVING NEEDS",
    # "ORAL HYGIENE",
    # "HAND & BODY",
    # "HOME DIAGNOSTICS",
    # "PERSONAL CLEANSING",
    # "ACNE/HSC",
    # "DIET/NUTRITION",
    # "DIGESTIVE HEALTH",
    # "COLD REMEDIES",
    # "SHAVING NEEDS",
    # "FACIAL CARE",
    # "EXTERNAL PAIN"
    # "ALLERGY REMEDIES",
    #     "ADULT CARE",
    # "FIRST AID",
    #     "DIGESTIVE HEALTH",
    #     "EXTERNAL PAIN",
    #     "DIET/NUTRITION",
    #     "STATIONERY",
    #     "BEVERAGES",
    #     "CHILDRENS REMEDIES",
    #     "COLD REMEDIES"
    # "GROCERY",
    # "HAIR CARE",
    "CANDY",
    # "TRIAL TRAVEL",
    # "DEODORANT"
    # "DEODORANTS",
    # "STATIONERY"
    # "FEMININE CARE",
    # "TRIAL TRAVEL",
    # "ORAL HYGIENE",
    # "HAIR COLOR"
    # "EXTERNAL PAIN",
    # "SNACKS",
    # "VITAMINS",
    # "GROCERY"
    # "ADULT CARE",
    # "FEMININE CARE",
    # "DEODORANTS",
    # "FIRST AID"
    # "HOUSEHOLD PAPER",
    # "ORAL HYGIENE",
    # "COLD BEVERAGES"
    # "COLD REMEDIES",
    # "BEVERAGES"
    # "ADULT CARE",
    # "COLD REMEDIES",
    # "HAIR COLOR",
    # "DEODORANTS",
    # "TEXTURED HAIR",
    # "STATIONERY",
    # "SNACKS"
    # "DIGESTIVE HEALTH",
    # "TRIAL TRAVEL",
    # "FIRST AID",
]

# Convert the store list into a comma-separated string (no quotes needed for numeric IDs)
excluded_stores_str = ", ".join(str(store) for store in excluded_stores)

# Convert the category lists into comma-separated strings with each category in quotes
wave_1_cat_str = ", ".join(f"'{cat}'" for cat in wave_1_cat)
wave_2_cat_str = ", ".join(f"'{cat}'" for cat in wave_2_cat)
wave_other_str = ", ".join(f"'{cat}'" for cat in wave_other)

# Example: below we only query Wave 1 categories.
# If you need both Wave 1 and Wave 2, you can either:
#   1) Use OR: AND (cat_dsc IN ({wave_1_cat_str}) OR cat_dsc IN ({wave_2_cat_str}))
#   2) Combine the two lists into one wave_12_cat = wave_1_cat + wave_2_cat
SQL_QUERY_TXN = f"""
SELECT 
    S.SKU_NBR,
    S.STORE_NBR, 
    S.CAT_DSC,
    S.TOTAL_SALES,
    NS.NEED_STATE
FROM 
    DL_FSCA_SLFSRV.TWA07.c830557_localization_last_yr_sales AS S
INNER JOIN 
    DL_FSCA_SLFSRV.TWA07.NEED_STATES_20250414_AM AS NS
    ON S.SKU_NBR = NS."PRODUCT_ID"
WHERE 
    state_cd NOT IN ('HI','PR')
    AND facility_typ_dsc = 'Retail CVS/pharmacy'
    AND str_actv_flg = 'ACTIVE'
    AND cat_dsc != 'COMPANY DEFAULT'
    AND total_sales > 0
    AND retail_sq_ft > 0
    AND cat_dsc IN ({wave_other_str})
    AND S.store_nbr NOT IN ({excluded_stores_str});
"""

# Read into a DataFrame
df_need_states = pd.read_sql(SQL_QUERY_TXN, conn)

# If needed, you can save to Parquet (optional)
# df_need_states.to_parquet("/home/jovyan/older code/notebooks/need_states_sales_31012025.parquet", index=False)


assert df_need_states.duplicated().sum() == 0, "duplicated values found"


df_need_states.shape


df_need_states["CAT_DSC"].unique()


df_need_states["CAT_DSC"].nunique()


df_need_states.query("NEED_STATE == 'light_bladder_leakage_-_fem_care_3'")


import pandas as pd

# 1. Identify the columns we need to group on (all except NEED_STATE)
group_cols = [col for col in df_need_states.columns if col != "NEED_STATE"]

# 2. Find the size of each group that shares everything but NEED_STATE
df_need_states["group_count"] = df_need_states.groupby(group_cols)["NEED_STATE"].transform("count")

# 3. Create the TOTAL_SALES_FINAL column
df_need_states["TOTAL_SALES_FINAL"] = df_need_states["TOTAL_SALES"] / df_need_states["group_count"]

# 4. Optionally, drop the helper column
# df_need_states.drop(columns='group_count', inplace=True)


df_need_states["CAT_DSC"].unique()


df_need_states_final = df_need_states[
    ["SKU_NBR", "STORE_NBR", "CAT_DSC", "NEED_STATE", "TOTAL_SALES_FINAL"]
]
df_need_states_final.columns = ["SKU_NBR", "STORE_NBR", "CAT_DSC", "NEED_STATE", "TOTAL_SALES"]
df_need_states_final.head()


df_need_states_final["TOTAL_SALES"].sum()

# 1,225,607,239.1799994
# 5,455,114,113.560007
# 692,326,367.4799997
# 4,617,713,522.709991
# 2,740,143,565.7900066
# 2,751,485,934.5999966
# 2,745,893,614.490009
# 2745893614.4899974
# 2718486555.750002 Feb 19
# 2718486555.750001 Feb 19 PC fix
# 2718486555.750006 Feb 19 OG & Rest
# 2718486555.750005 Feb 20
# 2718486555.7499933 Feb 25th
# 2716673750.6699977 feb 26th
# 2716673750.669995 feb 27th

# Wave 2 sales $1,854,860,922
# 1,777,631,228.34 feb 26th
# 1949867641.009997 feb 26th St

# Other
# 1,975,470,542 Mar 3rd

# Mar 27th 3,815,090,818


df_need_states_final["STORE_NBR"].nunique()
# 6898
# 6,890
# 6,898
# 6,898

# Wave 2 stores 6898


# df_need_states_final.query("NEED_STATE == 'body_lotion_41'")


df_need_states_final.shape
# Mar 17th (28888403, 5) 14 cats


# df_need_states_final.query("CAT_DSC == 'PERSONAL CLEANSING'").NEED_STATE.unique()


df_need_states_final = df_need_states_final.dropna(subset=["TOTAL_SALES"])


df_need_states_final.shape
# (4172014, 5)
# (31192266, 5) All Others Mar 7
# (41243738, 5) ALL Mar 4th
# (24952139, 5) Wave 1 Feb 19th
# (18745410, 5) Wave 1 Feb 19th PC fix
# (24919068, 5) Wave 1 feb 26th
# (24919068, 5) feb 27

# Wave 2
# (13906073, 5) feb 26th
# (16671884, 5) feb 26th St

# Other
# (28888403, 5) March 17th 14 cat


# df_need_states_final.to_parquet("/home/jovyan/older code/notebooks/need_states_sales_20250321.parquet", index=False)
df_need_states_final.to_parquet(
    "/home/jovyan/fsassortment/store_clustering/data/need_states_sales_20250414_AM.parquet",
    index=False,
)
