"""
internal_config.py

Configuration file for the internal data processing pipeline. 
Holds directory paths, clustering hyperparameters, categorical encoding details,
and any other relevant constants.
"""

# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------
DATA_DIR = "/home/jovyan/fsassortment/store_clustering/data/"
PILOT_CATEGORIES = "pilot_categories_07022025.xlsx"
PLANOGRAM = "pilot_categories_07022025.xlsx"
NEED_STATE_PARQUET = "need_states_sales_20250414_AM.parquet"

# --------------------------------------------------------------------------------
# Categoricals (for transformations)
# --------------------------------------------------------------------------------
cat_onehot = []
cat_ordinal = []

# --------------------------------------------------------------------------------
# Clustering & Preprocessing Hyperparameters
# --------------------------------------------------------------------------------
min_n_clusters_to_try = 3
max_n_clusters_to_try = 10
small_cluster_threshold = 20
pca_n_features_threshold = 20
max_n_pca_features = 15
cumulative_variance_target = 0.95
random_state = 9999
max_iter = 500
corr_threshold = 0.80

# --------------------------------------------------------------------------------
# Internal Condition Labels (INTERNAL_CDTS)
# --------------------------------------------------------------------------------
INTERNAL_CDTS = [
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
    # "HAIR CARE"
    # "EXTERNAL PAIN"
# "GROCERY",
# "HAIR CARE",
"CANDY",
# "TRIAL TRAVEL",
# "FEMININE CARE",
# "EXTERNAL PAIN", 
# "SNACKS",
# "GROCERY"
# "FEMININE CARE",
# "TRIAL TRAVEL",
# "ORAL HYGIENE",
# "HAIR COLOR"
# "HOUSEHOLD PAPER",
# "ADULT CARE",
# "COLD REMEDIES",
# "HAIR COLOR",
# "DEODORANTS",
# "TEXTURED HAIR",
# "STATIONERY", 
# "SNACKS"
# "BEVERAGES"  
# "EXTERNAL PAIN"
# "DIGESTIVE HEALTH",
# "TRIAL TRAVEL",
# "FIRST AID",
]


INTERNAL_CLUSTERING_OUTPUT_DIR = "/home/jovyan/fsassortment/store_clustering/internal_content"
