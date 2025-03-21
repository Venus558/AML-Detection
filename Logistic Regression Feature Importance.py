import pickle
import pandas as pd

# Load the best logistic regression model from the pickle file
with open('best_logistic_model.pkl', 'rb') as f:
    best_logreg_model = pickle.load(f)

# Load the feature names from the saved pkl file
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

# Get the model coefficients
coefficients = best_logreg_model.coef_[0]

# Create a DataFrame to show the feature importance
logreg_importance_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
})

# Calculate the absolute importance for better interpretation and divide by 100
logreg_importance_df['Importance'] = logreg_importance_df['Coefficient'].abs() / 100
logreg_importance_df['Coefficient'] = logreg_importance_df['Coefficient'] / 100

# Sort by absolute importance
logreg_importance_df = logreg_importance_df.sort_values(by='Importance', ascending=False)

# Save the result as a pickle file
logreg_importance_df.to_pickle('Logistic Regression Feature Importance.pkl')

# Print the DataFrame to see the result
print(logreg_importance_df)
