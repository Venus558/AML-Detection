import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load the XGBoost model
with open("best_xgboost_model.pkl", "rb") as xgb_file:
    best_xgb_model = pickle.load(xgb_file)

# Extract feature importances
feature_importances = best_xgb_model.feature_importances_

# Step 1: Load the column names from the pickle file
with open('features.pkl', 'rb') as f:
    features = pd.read_pickle(f)

# Create a DataFrame for feature importances
xgb_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
xgb_importance_df = xgb_importance_df.sort_values(by='Importance', ascending=False)

xgb_importance_df.to_pickle('XGBoost Feature Importance.pkl')

# Display the feature importances
print("XGBoost Feature Importances:")
print(xgb_importance_df)

