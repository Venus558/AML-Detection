import pandas as pd

# Load the dataset
df = pd.read_csv("Financial Data Set.csv")

# Convert 'type' column to category
df['type'] = df['type'].astype('category')

# Check memory usage before and after conversion
print(f"Memory usage before conversion: {df.memory_usage(deep=True).sum()} bytes")
print(f"Memory usage after conversion: {df.memory_usage(deep=True).sum()} bytes")

# Check the data types again to confirm the change
print("\nUpdated data types:")
print(df.dtypes)

# Save the DataFrame as a pickle file
df.to_pickle("Financial_Data_Set.pkl")