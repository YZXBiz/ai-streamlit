"""
config.py

Central configuration file for Clustering_Repo_CVS.
Holds directory paths, lists of columns, numeric thresholds, etc.
"""

import os

# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------

DATA_DIR = "/home/jovyan/fsassortment/store_clustering/data/"
CATEGORY_MAPPING_DIR = "/home/jovyan/fsassortment/store_clustering/data/category_dmm_mapping.xlsx"


# --------------------------------------------------------------------------------
# Categoricals (if needed for tansformations)
# --------------------------------------------------------------------------------
cat_onehot = []
cat_ordinal = []

# --------------------------------------------------------------------------------
# Clustering & Preprocessing Hyperparameters
# --------------------------------------------------------------------------------
min_n_clusters_to_try = 7
max_n_clusters_to_try = 15
small_cluster_threshold = 40
pca_n_features_threshold = 20
max_n_pca_features = 15
cumulative_variance_target = 0.95
random_state = 9999
max_iter = 500
corr_threshold = 0.80

# --------------------------------------------------------------------------------
# Columns to Drop from Analysis
# --------------------------------------------------------------------------------
DROP_COLS = [
    # "WHITE_PCT_PZ_PG",
    # "BLACK_PCT_PZ_PG",
    # "ASIAN_PCT_PZ_PG",
    # "HISP_PCT_PZ_PG",
    # "OTHER_RACE_PCT_PZ_PG",
    "WALM_CNT_MEAN",
    "TRGT_CNT_MEAN"
]

# --------------------------------------------------------------------------------
# Merchandising Groups (In Scope)
# --------------------------------------------------------------------------------
MERCH_GROUP_INSCOPE = [
    "CONSUMER HEALTH CARE",
    "EDIBLES",
    "BEAUTY CARE",
    "PERSONAL CARE",
    "GENERAL MERCHANDISE"
]

# --------------------------------------------------------------------------------
# Competitive Drive Columns
# --------------------------------------------------------------------------------
COMP_DRIVE_COLS = [
    "STORE_NBR",
    "mins_albertson_nearest_1",
    "mins_traderjoes_nearest_1",
    "mins_walgreens_nearest_1",
    "mins_walmart_nearest_1",
    "mins_wegmans_nearest_1",
    "mins_weismarket_nearest_1",
    "mins_wholefoods_nearest_1",
    "mins_dollartree_nearest_1",
    "mins_costco_nearest_1",
    "mins_DG_nearest_1",
    "mins_foodlion_nearest_1",
    "mins_heb_nearest_1",
    "mins_kroger_nearest_1",
    "mins_publix_nearest_1",
    "mins_riteaid_nearest_1",
    "mins_target_nearest_1"
]

# --------------------------------------------------------------------------------
# Spending Columns
# --------------------------------------------------------------------------------
SPEND_COLS = [
    "STORE_NBR",
    "2024 Average Alcoholic beverages",
    "2024 Average Apparel and services",
    "2024 Average Cash contributions",
    "2024 Average Education",
    "2024 Average Entertainment",
    "2024 Average Food",
    "2024 Average Housing",
    "2024 Average Miscellaneous",
    "2024 Average Personal care products and services",
    "2024 Average Personal insurance and pensions",
    "2024 Average Reading",
    "2024 Average Tobacco products and smoking supplies",
    "2024 Average Transportation"
]

# --------------------------------------------------------------------------------
# Co-Tenant Columns
# --------------------------------------------------------------------------------
COTENANT_COLS = [
    "STORE_NBR",
    "tenant_cnt"
]

# --------------------------------------------------------------------------------
# Competitor Columns (Totals)
# --------------------------------------------------------------------------------
COMP_COLS = [
    "STORE_NBR",
    "Total Target",
    "Total Trader Joes",
    "Total Walgreens",
    "Total Whole Foods",
    "Total Albertsons",
    "Total Costco",
    "Total Dollar General",
    "Total Food Lion",
    "Total H-E-B",
    "Total Kroger",
    "Total Publix",
    "Total Rite Aid",
    "Total Weis Markets",
    "Total DollarTree",
    "Total Walmart",
    "Total Wegmans"
]

# --------------------------------------------------------------------------------
# Columns from Placer.ai Data
# --------------------------------------------------------------------------------
PLACER_COLS = [
    "STORE_NBR",
    # Dwell time
    "avg_dwell_time",
    "avg_visits_per_day",
    "median_dwell_time",
    "median_visits_per_day",
    # "census_captured_avg_hhi",

    # Education distribution
    "census_captured_by_education_elementary",
    "census_captured_by_education_high_school",
    "census_captured_by_education_college_or_associates_degree",
    "census_captured_by_education_bachelors_degree",
    "census_captured_by_education_advanced_degree",

    # Household Income distribution
    "census_captured_by_hhi_less_than_10K",
    "census_captured_by_hhi_10K_to_15K",
    "census_captured_by_hhi_15K_to_20K",
    "census_captured_by_hhi_20K_to_25K",
    "census_captured_by_hhi_25K_to_30K",
    "census_captured_by_hhi_30K_to_35K",
    "census_captured_by_hhi_35K_to_40K",
    "census_captured_by_hhi_40K_to_45K",
    "census_captured_by_hhi_45K_to_50K",
    "census_captured_by_hhi_50K_to_60K",
    "census_captured_by_hhi_60K_to_75K",
    "census_captured_by_hhi_75K_to_100K",
    "census_captured_by_hhi_100K_to_125K",
    "census_captured_by_hhi_125K_to_150K",
    "census_captured_by_hhi_150K_to_200K",
    "census_captured_by_hhi_more_than_200K",

    # Foot traffic
    "foottraffic",
    "foottraffic_per_sqft",
    "unique_visitors"
]

# --------------------------------------------------------------------------------
# Additional Columns
# --------------------------------------------------------------------------------
ADDITIONAL_COLS = [
    "STORE_NBR",
    "DAYTIME_POP_CNT",
    "DAYNITE_RATIO_FCTR_NBR"
]

# --------------------------------------------------------------------------------
# Core Columns for Clustering
# --------------------------------------------------------------------------------
core_cols = [
    "STORE_NBR",
    # "TRAFFIC_CNT",
    "FS_SALES_PER_SQFT",
    "SEC_CITY_PCT",
    "TOWN_AND_RURAL_PCT",
    "RENTERS_PCT",
    "URBAN_PCT",
    "SUBURBAN_PCT",
    "HH_W_INCOME_0_TO_15000_PCT_MEAN",
    "HH_W_INCOME_15000_TO_25000_PCT_MEAN",
    "HH_W_INCOME_25000_TO_35000_PCT_MEAN",
    "HH_W_INCOME_35000_TO_50000_PCT_MEAN",
    "HH_W_INCOME_50000_TO_75000_PCT_MEAN",
    "HH_W_INCOME_75000_TO_100000_PCT_MEAN",
    "HH_W_INCOME_100000_TO_150000_PCT_MEAN",
    "HH_W_INCOME_150000_AND_UP_PCT_MEAN",
    "POP_0_TO_4_PCT_MEAN",
    "POP_5_TO_14_PCT_MEAN",
    "POP_15_TO_24_PCT_MEAN",
    "POP_25_TO_34_PCT_MEAN",
    "POP_35_TO_44_PCT_MEAN",
    "POP_45_TO_54_PCT_MEAN",
    "POP_55_TO_64_PCT_MEAN",
    "POP_65_TO_74_PCT_MEAN",
    "POP_75_AND_UP_PCT_MEAN",
    "LATITUDE",
    "LONGITUDE",
    "PHARM_RATIO",
    "HOME_VALUE_LESS_THAN_40000_PCT_MEAN",
    "HOME_VALUE_40000_TO_59999_PCT_MEAN",
    "HOME_VALUE_60000_TO_79999_PCT_MEAN",
    "HOME_VALUE_80000_TO_99999_PCT_MEAN",
    "HOME_VALUE_100000_TO_149999_PCT_MEAN",
    "HOME_VALUE_150000_TO_199999_PCT_MEAN",
    "HOME_VALUE_200000_TO_299999_PCT_MEAN",
    "HOME_VALUE_300000_TO_399999_PCT_MEAN",
    "HOME_VALUE_400000_TO_499999_PCT_MEAN",
    "HOME_VALUE_500000_OR_MORE_PCT_MEAN",

    # From Placer.ai data
    "avg_dwell_time",
    "census_captured_by_education_elementary",
    "census_captured_by_education_high_school",
    "census_captured_by_education_college_or_associates_degree",
    "census_captured_by_education_bachelors_degree",
    "census_captured_by_education_advanced_degree",
    "foottraffic",
    "unique_visitors",
    "tenant_cnt",
    "DAYTIME_POP_CNT",
    "DAYNITE_RATIO_FCTR_NBR",
]

# --------------------------------------------------------------------------------
# Category-Specific Variable Lists
# --------------------------------------------------------------------------------
BEAUTY = [
    "2024 Average Personal care products and services",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "CHAIN_DRUG_STR_NO_RX_CNT_MEAN",
    "MASS_MERCH_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "WHOLESALE_MBR_CNT_MEAN",
    "GROC_CNT_MEAN",
    "Total Walgreens",
    "Total Rite Aid",
    "Total Walmart",
    "Total Target",
    "Total Dollar General",
    "Total DollarTree",
    "Total Kroger",
    "Total Publix",
    "Total Albertsons",
    "Total H-E-B",
    "Total Food Lion",
    "Total Costco",
    "Total Trader Joes",
    "Total Wegmans",
    "Total Weis Markets",
    "Total Whole Foods"
]

ACUTE_HEALTHCARE = [
    "2024 Average Personal care products and services",
    "2024 Average Personal insurance and pensions",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "CHAIN_DRUG_STR_NO_RX_CNT_MEAN",
    "INDEP_CNT_MEAN",
    "GROC_PHARM_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "Total Walgreens",
    "Total Rite Aid",
    "Total Walmart",
    "Total Target",
    "Total Dollar General",
    "Total DollarTree",
    "Total Kroger",
    "Total Publix",
    "Total Albertsons",
    "Total H-E-B",
    "Total Food Lion",
    "Total Costco",
    "Total Trader Joes",
    "Total Wegmans",
    "Total Weis Markets",
    "Total Whole Foods"
]

WELLNESS = [
    "2024 Average Personal care products and services",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "CHAIN_DRUG_STR_NO_RX_CNT_MEAN",
    "WHOLESALE_MBR_CNT_MEAN",
    "INDEP_CNT_MEAN",
    "GROC_CNT_MEAN",
    "MAJOR_GROC_CNT_MEAN",
    "MASS_MERCH_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "Total Walgreens",
    "Total Rite Aid",
    "Total Walmart",
    "Total Target",
    "Total Dollar General",
    "Total DollarTree",
    "Total Kroger",
    "Total Publix",
    "Total Albertsons",
    "Total H-E-B",
    "Total Food Lion",
    "Total Costco",
    "Total Trader Joes",
    "Total Wegmans",
    "Total Weis Markets",
    "Total Whole Foods"
]

EMPOWERED_AGING = [
    "2024 Average Personal care products and services",
    "2024 Average Personal insurance and pensions",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "CHAIN_DRUG_STR_NO_RX_CNT_MEAN",
    "MASS_MERCH_CNT_MEAN",
    "MAJOR_MASS_CNT_MEAN",
    "WHOLESALE_MBR_CNT_MEAN",
    "INDEP_CNT_MEAN",
    "GROC_PHARM_CNT_MEAN",
    "MAJOR_GROC_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "Total Walgreens",
    "Total Rite Aid",
    "Total Walmart",
    "Total Target",
    "Total Dollar General",
    "Total DollarTree",
    "Total Kroger",
    "Total Publix",
    "Total Albertsons",
    "Total H-E-B",
    "Total Food Lion",
    "Total Costco",
    "Total Trader Joes",
    "Total Wegmans",
    "Total Weis Markets",
    "Total Whole Foods"
]

CONSUMABLES = [
    "MAJOR_GROC_CNT_MEAN",
    "GROC_CNT_MEAN",
    "WHOLESALE_MBR_CNT_MEAN",
    "MASS_MERCH_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "Total Walmart",
    "Total Kroger",
    "Total Publix",
    "Total Food Lion",
    "Total H-E-B",
    "Total Wegmans",
    "Total Trader Joes",
    "Total Albertsons",
    "Total Weis Markets",
    "Total Whole Foods",
    "Total Target",
    "Total Costco",
    "Total Dollar General",
    "Total DollarTree",
    "Total Walgreens",
    "Total Rite Aid",
    "2024 Average Food",
    "2024 Average Miscellaneous",
    "2024 Average Alcoholic beverages"
]

HOUSEHOLD = [
    "2024 Average Miscellaneous",
    "2024 Average Housing",
    "MASS_MERCH_CNT_MEAN",
    "GROC_CNT_MEAN",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "WHOLESALE_MBR_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "Total Walmart",
    "Total Target",
    "Total Dollar General",
    "Total DollarTree",
    "Total Walgreens",
    "Total Rite Aid",
    "Total Kroger",
    "Total Publix",
    "Total Albertsons",
    "Total H-E-B",
    "Total Food Lion",
    "Total Costco",
    "Total Trader Joes",
    "Total Wegmans",
    "Total Weis Markets",
    "Total Whole Foods"
]

SEASONAL_AND_GIFTING = [
    "2024 Average Apparel and services",
    "2024 Average Reading",
    "2024 Average Food",
    "2024 Average Entertainment",
    "MASS_MERCH_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "WHOLESALE_MBR_CNT_MEAN",
    "Total Target",
    "Total Walmart",
    "Total Dollar General",
    "Total DollarTree",
    "Total Costco",
    "Total Walgreens",
    "Total Rite Aid",
    "Total Kroger",
    "Total Publix",
    "Total H-E-B",
    "Total Food Lion",
    "Total Albertsons",
    "Total Trader Joes",
    "Total Wegmans",
    "Total Weis Markets",
    "Total Whole Foods"
]

PERSONAL_CARE = [
    "2024 Average Personal care products and services",
    "CHAIN_DRUG_STR_CNT_MEAN",
    "CHAIN_DRUG_STR_NO_RX_CNT_MEAN",
    "MASS_MERCH_CNT_MEAN",
    "FAMDOLGEN_CNT_MEAN",
    "Total Walgreens",
    "Total Rite Aid",
    "Total Walmart",
    "Total Target",
    "Total Dollar General",
    "Total DollarTree",
    "Total Kroger",
    "Total Publix",
    "Total Albertsons",
    "Total Costco"
]

COLUMNS_TO_DROP = [
    "TRAFFIC_CNT",
    'EDUC_0_TO_9TH_PCT_MEAN',
    'EDUC_SOME_HS_PCT_MEAN',
    'EDUC_HS_GRAD_PCT_MEAN',
    'EDUC_SOME_COLLEGE_PCT_MEAN',
    'EDUC_ASSOCIATE_DEGREE_PCT_MEAN',
    'EDUC_BACHELOR_DEGREE_PCT_MEAN',
    'EDUC_GRAD_OR_PROF_DEGREE_PCT_MEAN',
    
    'HOME_VALUE_MEDIAN_AMT_MEAN',
 
    'AVERAGE_INCOME_PER_PERSON',
    'AVERAGE_AGE',
    'AVERAGE_PERSONS_PER_HH',
     
    'median_dwell_time',
    'median_visits_per_day',
    'foottraffic_per_sqft',
    'census_captured_by_hhi_less_than_10K',
    'census_captured_by_hhi_10K_to_15K',
    'census_captured_by_hhi_15K_to_20K',
    'census_captured_by_hhi_20K_to_25K',
    'census_captured_by_hhi_25K_to_30K',
    'census_captured_by_hhi_30K_to_35K',
    'census_captured_by_hhi_35K_to_40K',
    'census_captured_by_hhi_40K_to_45K',
    'census_captured_by_hhi_45K_to_50K',
    'census_captured_by_hhi_50K_to_60K',
    'census_captured_by_hhi_60K_to_75K',
    'census_captured_by_hhi_75K_to_100K',
    'census_captured_by_hhi_100K_to_125K',
    'census_captured_by_hhi_125K_to_150K',
    'census_captured_by_hhi_150K_to_200K',
    'census_captured_by_hhi_more_than_200K',
    
    "2024 Average Tobacco products and smoking supplies"]

# Ranking

# Store / Target columns
STORE_COL = "STORE_NBR"
TARGET_COL = "cluster_label"

# Columns to ignore for modeling
COLUMNS_TO_IGNORE = []

# Train/test split parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# LightGBM hyperparameters
NUM_LEAVES = 31
MAX_DEPTH = -1
N_ESTIMATORS = 100
LEARNING_RATE = 0.1
STOPPING_ROUNDS = 10

# For SHAP or other feature importance
TOP_N_FEATURES = 20  # Example if you want a default for top_n

RANKING_OUTPUT_DIR = "/home/jovyan/fsassortment/store_clustering/feature_ranking_results"

RENAME_DICT = {
    "STORE_NBR": "Store #",
    "TRAFFIC_CNT": "Traffic Count",
    "FS_SALES_PER_SQFT": "Front Store Sales/sqft",
    "RENTERS_PCT": "% Renters",
    "URBAN_PCT": "% Urban",
    "SUBURBAN_PCT": "% Suburban",
    "SEC_CITY_PCT": "% Secondary City",
    "TOWN_AND_RURAL_PCT": "% Town/Rural",
    "KMART_CNT_MEAN": "Avg # Kmart",
    "CHAIN_DRUG_STR_CNT_MEAN": "Avg # Chain Drug Stores",
    "CHAIN_DRUG_STR_NO_RX_CNT_MEAN": "Avg # Chain Drug (No Rx)",
    "MASS_MERCH_CNT_MEAN": "Avg # Mass Merch",
    "MAJOR_MASS_CNT_MEAN": "Avg # Major Mass Merch",
    "WHOLESALE_MBR_CNT_MEAN": "Avg # Wholesale Mem",
    "INDEP_CNT_MEAN": "Avg # Independents",
    "GROC_CNT_MEAN": "Avg # Grocery",
    "GROC_PHARM_CNT_MEAN": "Avg # Grocery Pharm",
    "MAJOR_GROC_CNT_MEAN": "Avg # Major Grocery",
    "FAMDOLGEN_CNT_MEAN": "Avg # Fam/Dollar/Gen",
    "HH_W_INCOME_0_TO_15000_PCT_MEAN": "% HH Inc $0-15k",
    "HH_W_INCOME_15000_TO_25000_PCT_MEAN": "% HH Inc $15-25k",
    "HH_W_INCOME_25000_TO_35000_PCT_MEAN": "% HH Inc $25-35k",
    "HH_W_INCOME_35000_TO_50000_PCT_MEAN": "% HH Inc $35-50k",
    "HH_W_INCOME_50000_TO_75000_PCT_MEAN": "% HH Inc $50-75k",
    "HH_W_INCOME_75000_TO_100000_PCT_MEAN": "% HH Inc $75-100k",
    "HH_W_INCOME_100000_TO_150000_PCT_MEAN": "% HH Inc $100-150k",
    "HH_W_INCOME_150000_AND_UP_PCT_MEAN": "% HH Inc $150k+",
    "POP_0_TO_4_PCT_MEAN": "% Age 0-4",
    "POP_5_TO_14_PCT_MEAN": "% Age 5-14",
    "POP_15_TO_24_PCT_MEAN": "% Age 15-24",
    "POP_25_TO_34_PCT_MEAN": "% Age 25-34",
    "POP_35_TO_44_PCT_MEAN": "% Age 35-44",
    "POP_45_TO_54_PCT_MEAN": "% Age 45-54",
    "POP_55_TO_64_PCT_MEAN": "% Age 55-64",
    "POP_65_TO_74_PCT_MEAN": "% Age 65-74",
    "POP_75_AND_UP_PCT_MEAN": "% Age 75+",
    "LATITUDE": "Latitude",
    "LONGITUDE": "Longitude",
    "PHARM_RATIO": "Pharmacy Ratio",
    "EDUC_0_TO_9TH_PCT_MEAN": "% Edu 0-9th",
    "EDUC_SOME_HS_PCT_MEAN": "% Edu Some HS",
    "EDUC_HS_GRAD_PCT_MEAN": "% Edu HS Grad",
    "EDUC_SOME_COLLEGE_PCT_MEAN": "% Edu Some College",
    "EDUC_ASSOCIATE_DEGREE_PCT_MEAN": "% Edu Assoc Deg",
    "EDUC_BACHELOR_DEGREE_PCT_MEAN": "% Edu Bachelor's Deg",
    "EDUC_GRAD_OR_PROF_DEGREE_PCT_MEAN": "% Edu Grad/Prof Deg",
    "HOME_VALUE_LESS_THAN_40000_PCT_MEAN": "% Home Value <40k",
    "HOME_VALUE_40000_TO_59999_PCT_MEAN": "% Home Value 40-60k",
    "HOME_VALUE_60000_TO_79999_PCT_MEAN": "% Home Value 60-80k",
    "HOME_VALUE_80000_TO_99999_PCT_MEAN": "% Home Value 80-100k",
    "HOME_VALUE_100000_TO_149999_PCT_MEAN": "% Home Value 100-150k",
    "HOME_VALUE_150000_TO_199999_PCT_MEAN": "% Home Value 150-200k",
    "HOME_VALUE_200000_TO_299999_PCT_MEAN": "% Home Value 200-300k",
    "HOME_VALUE_300000_TO_399999_PCT_MEAN": "% Home Value 300-400k",
    "HOME_VALUE_400000_TO_499999_PCT_MEAN": "% Home Value 400-500k",
    "HOME_VALUE_500000_OR_MORE_PCT_MEAN": "% Home Value 500k+",
    "HOME_VALUE_MEDIAN_AMT_MEAN": "Median Home Value",
    "2024 Average Alcoholic beverages": "2024 Alcoholic Bev ($)",
    "2024 Average Apparel and services": "2024 Apparel/Services ($)",
    "2024 Average Cash contributions": "2024 Cash Contrib ($)",
    "2024 Average Education": "2024 Education ($)",
    "2024 Average Entertainment": "2024 Entertainment ($)",
    "2024 Average Food": "2024 Food ($)",
    "2024 Average Housing": "2024 Housing ($)",
    "2024 Average Miscellaneous": "2024 Misc ($)",
    "2024 Average Personal care products and services": "2024 Personal Care ($)",
    "2024 Average Personal insurance and pensions": "2024 Insurance/Pension ($)",
    "2024 Average Reading": "2024 Reading ($)",
    "2024 Average Tobacco products and smoking supplies": "2024 Tobacco ($)",
    "2024 Average Transportation": "2024 Transport ($)",
    "tenant_cnt": "# Tenants",
    "Total Target": "# Target",
    "Total Trader Joes": "# Trader Joe's",
    "Total Walgreens": "# Walgreens",
    "Total Whole Foods": "# Whole Foods",
    "Total Albertsons": "# Albertsons",
    "Total Costco": "# Costco",
    "Total Dollar General": "# Dollar General",
    "Total Food Lion": "# Food Lion",
    "Total H-E-B": "# H-E-B",
    "Total Kroger": "# Kroger",
    "Total Publix": "# Publix",
    "Total Rite Aid": "# Rite Aid",
    "Total Weis Markets": "# Weis",
    "Total DollarTree": "# Dollar Tree",
    "Total Walmart": "# Walmart",
    "Total Wegmans": "# Wegmans",
    "DAYTIME_POP_CNT": "Daytime Pop",
    "DAYNITE_RATIO_FCTR_NBR": "Day/Night Ratio",
    "avg_dwell_time": "Avg Dwell Time",
    "avg_visits_per_day": "Avg Visits/Day",
    "census_captured_by_education_elementary": "Elem Educ",
    "census_captured_by_education_high_school": "High Sch Educ",
    "census_captured_by_education_college_or_associates_degree": "College/Assoc Educ",
    "census_captured_by_education_bachelors_degree": "Bachelor's Educ",
    "census_captured_by_education_advanced_degree": "Adv Educ",
    "census_captured_by_hhi_less_than_10K": "HHI <10k",
    "census_captured_by_hhi_10K_to_15K": "HHI 10-15k",
    "census_captured_by_hhi_15K_to_20K": "HHI 15-20k",
    "census_captured_by_hhi_20K_to_25K": "HHI 20-25k",
    "census_captured_by_hhi_25K_to_30K": "HHI 25-30k",
    "census_captured_by_hhi_30K_to_35K": "HHI 30-35k",
    "census_captured_by_hhi_35K_to_40K": "HHI 35-40k",
    "census_captured_by_hhi_40K_to_45K": "HHI 40-45k",
    "census_captured_by_hhi_45K_to_50K": "HHI 45-50k",
    "census_captured_by_hhi_50K_to_60K": "HHI 50-60k",
    "census_captured_by_hhi_60K_to_75K": "HHI 60-75k",
    "census_captured_by_hhi_75K_to_100K": "HHI 75-100k",
    "census_captured_by_hhi_100K_to_125K": "HHI 100-125k",
    "census_captured_by_hhi_125K_to_150K": "HHI 125-150k",
    "census_captured_by_hhi_150K_to_200K": "HHI 150-200k",
    "census_captured_by_hhi_more_than_200K": "HHI 200k+",
    "foottraffic": "Foot Traffic",
    "foottraffic_per_sqft": "Foot Traffic/sqft",
    "unique_visitors": "Unique Visitors"
}

# Add a config variable for the base feature_ranking_results directory:
FEATURE_RANKING_RESULTS_DIR = "/home/jovyan/fsassortment/store_clustering/feature_ranking_results"