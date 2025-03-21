import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix

# Load the preprocessed (scaled) data
with open("scaled_data.pkl", "rb") as file:
    X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(file)

# Load the best models
with open('best_xgboost_model.pkl', 'rb') as f:
    best_xgb_model = pickle.load(f)

with open('best_logistic_model.pkl', 'rb') as f:
    best_logistic_model = pickle.load(f)

# Make predictions using the best models
xgb_preds = best_xgb_model.predict(X_test_scaled)
logreg_preds = best_logistic_model.predict(X_test_scaled)

# Generate confusion matrix for both models
xgb_conf_matrix = confusion_matrix(y_test, xgb_preds)
logreg_conf_matrix = confusion_matrix(y_test, logreg_preds)

# Create a DataFrame with the labeled confusion matrix
conf_matrix_df = pd.DataFrame({
    'Metric': ['True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)', 'True Positive (TP)'],
    'XGBoost Confusion Matrix': xgb_conf_matrix.flatten(),
    'Logistic Regression Confusion Matrix': logreg_conf_matrix.flatten()
})

# Save the confusion matrix as a CSV
conf_matrix_df.to_csv('confusion_matrix_with_labels.csv', index=False)
