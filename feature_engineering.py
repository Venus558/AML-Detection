import pandas as pd
import pickle

# Load the dataset from pickle
with open("Financial_Data_Set.pkl", "rb") as file:
    df = pickle.load(file)

# Create 'sender_percentage_withdrawn' column (percentage withdrawn from sender's account)
df['sender_percentage_withdrawn'] = df['amount'] / df['oldbalanceOrg'] * 100

# Create 'recipient_percentage_increased' column (percentage increase to recipient's account)
df['recipient_percentage_increased'] = df['amount'] / df['oldbalanceDest'] * 100

# Check for any infinite values in these new columns (divide by 0 scenario)
df['sender_percentage_withdrawn'].replace([float('inf'), -float('inf')], float('nan'), inplace=True)
df['recipient_percentage_increased'].replace([float('inf'), -float('inf')], float('nan'), inplace=True)

# Show the first few rows to verify the new columns
print(df.head())

# Save the dataframe back to a pickle file
with open("processed_data.pkl", "wb") as file:
    pickle.dump(df, file)
