import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the processed data after featured engineering (from the Feature Engineering file)
with open("processed_data.pkl", "rb") as file:
    df = pickle.load(file)

# Prepare the features (X) and target (y)
X = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
        'newbalanceDest', 'sender_percentage_withdrawn', 'recipient_percentage_increased']]

y = df['isFraud']

# Handle missing values
X.fillna(0, inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save as a Pickle
with open("scaled_data.pkl", "wb") as file:
    pickle.dump([X_train_scaled, X_test_scaled, y_train, y_test], file)

# Extract the column names from X
column_names = X.columns.tolist()

# Save the column names to a pickle file
with open('features.pkl', 'wb') as f:
    pd.to_pickle(column_names, f)
