# check_data.py

import pandas as pd

# Cek file CSV
csv_path = "data\processed\[Preprocessed]Plantar Thermogram Data Analysis.csv"
print(f"Checking file: {csv_path}\n")

try:
    df = pd.read_csv(csv_path)
    print("Columns found:")
    print(df.columns.tolist())
    print(f"\nShape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
except Exception as e:
    print(f"Error reading file: {e}")