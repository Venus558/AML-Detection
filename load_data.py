import pandas as pd  

# Load the dataset
df = pd.read_csv("Financial Data Set.csv")

# Show first 5 rows
print(df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Check data types and basic info
print("\nDataset Info:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())