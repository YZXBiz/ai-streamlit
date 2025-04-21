import pickle
import os
import pandas as pd

# Path to the pickle file
pickle_path = '/workspaces/clustering-dagster/data/internal/cluster_assignments.pkl'

print(f"Checking pickle file at: {pickle_path}")
print(f"File exists: {os.path.exists(pickle_path)}")
print(f"File size: {os.path.getsize(pickle_path)} bytes")

try:
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
        
    print('\nSuccessfully loaded pickle file')
    print('Type:', type(data))

    if hasattr(data, '__len__'):
        print('Length:', len(data))
    else:
        print('No length attribute')
        
    # Show a sample or summary
    if isinstance(data, pd.DataFrame):
        print('\nDataFrame info:')
        print(data.info())
        print('\nFirst 5 rows:')
        print(data.head(5))
    elif isinstance(data, dict):
        print('\nDictionary keys:', list(data.keys()))
        for key, value in data.items():
            print(f'\nKey: {key}, Type: {type(value)}')
            if isinstance(value, pd.DataFrame):
                print(f'DataFrame shape: {value.shape}')
                print(f'Columns: {value.columns.tolist()}')
                print(f'First 3 rows:\n{value.head(3)}')
    else:
        print('\nContent:', data)
except Exception as e:
    print(f"\nError loading pickle file: {e}") 