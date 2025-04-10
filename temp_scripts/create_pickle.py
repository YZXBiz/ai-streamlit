import pandas as pd
import polars as pl
import pickle

# Create sample data by category
categories = ['ADULT CARE', 'BABY CARE', 'NUTRITION', 'BEAUTY', 'PERSONAL CARE', 'MEDICATION', 'HOME CARE']
data_by_category = {}

for category in categories:
    # Create a sample DataFrame for each category
    df = pl.DataFrame({
        'SKU_NBR': list(range(1000 + categories.index(category) * 3, 1000 + categories.index(category) * 3 + 3)),
        'STORE_NBR': [101, 102, 103],
        'SALES': [1000 * (categories.index(category) + 1) + i * 500 for i in range(3)],
        'UNITS': [100 * (categories.index(category) + 1) + i * 50 for i in range(3)],
        'PRICE': [10 + categories.index(category) * 2 + i for i in range(3)]
    })
    
    # Store the DataFrame in the dictionary
    data_by_category[category] = df

# Save the dictionary to a pickle file
with open('/workspaces/testing-dagster/data/internal/sales_by_category.pkl', 'wb') as f:
    pickle.dump(data_by_category, f)

print('Created sales_by_category.pkl with sample data for categories:', categories) 