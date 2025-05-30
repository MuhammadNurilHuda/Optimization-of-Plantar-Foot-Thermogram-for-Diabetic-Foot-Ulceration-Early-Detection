# debug_csv.py

import pandas as pd
import os

csv_path = "data\processed\[Preprocessed]Plantar Thermogram Data Analysis.csv"

print(f"File exists: {os.path.exists(csv_path)}")
print(f"File path: {os.path.abspath(csv_path)}")

# Baca CSV dengan berbagai cara
try:
    # Method 1: Basic read
    df = pd.read_csv(csv_path)
    print("\n=== Method 1: Basic read ===")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    
    # Cek untuk spasi atau karakter tersembunyi
    print("\n=== Column names with quotes ===")
    for i, col in enumerate(df.columns):
        print(f"{i}: '{col}' (length: {len(col)})")
    
    # Cek apakah ada spasi di nama kolom
    print("\n=== Checking for spaces ===")
    for col in df.columns:
        if col.strip() != col:
            print(f"Column '{col}' has extra spaces!")
    
    # Print raw column names
    print("\n=== Raw column representation ===")
    print(repr(df.columns.tolist()))
    
except Exception as e:
    print(f"Error with basic read: {e}")

# Method 2: Read with different encoding
print("\n=== Trying different encodings ===")
for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-8-sig']:
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"\nSuccess with {encoding}:")
        print(f"Columns: {df.columns.tolist()}")
        break
    except Exception as e:
        print(f"Failed with {encoding}: {e}")

# Check first few rows
print("\n=== First 3 rows ===")
print(df.head(3))

# Check if 'Gender' column exists (case sensitive)
if 'Gender' in df.columns:
    print("\n✓ 'Gender' column EXISTS")
else:
    print("\n✗ 'Gender' column NOT FOUND")
    print("Available columns:")
    for col in df.columns:
        if 'gender' in col.lower():
            print(f"  - Found similar: '{col}'")