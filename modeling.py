import pickle
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the preprocessed (scaled) data
with open("scaled_data.pkl", "rb") as file:
    X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(file)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)

print(xgb_model)

# Set up hyperparameter grid for RandomizedSearchCV
param_dist = {
    "n_estimators": [100, 200, 300, 400, 500],  # Number of boosting (trees) rounds
    "learning_rate": [0.05, 0.1, 0.2, 0.3],     # Step size shrinkage to prevent overfitting
    "max_depth": [3, 5, 7, 9, 12],              # Maximum depth of each decision tree
    "min_child_weight": [1, 3, 5, 7],           # Minimum sum of instance weight (hessian) needed in a child
    "gamma": [0, 0.1, 0.2, 0.3, 0.4],           # Minimum loss reduction required to make a further partition (regularization)
    "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],     # Fraction of training data used for each tree (row sampling)
    "colsample_bytree": [0.5, 0.7, 0.8, 0.9, 1.0],  # Fraction of features (columns) used per tree
    "scale_pos_weight": [len(y_train[y_train == 0]) / len(y_train[y_train == 1])]  # Class imbalance ratio to handle skewed datasets
}

# Perform Randomized Search
random_search = RandomizedSearchCV(
    xgb_model, param_distributions=param_dist, 
    n_iter=10, cv=2, scoring="accuracy", n_jobs=-1, verbose=2, random_state=42
)

print("Starting hyperparameter tuning...")
random_search.fit(X_train_scaled, y_train)

# Get the best model
best_xgb_model = random_search.best_estimator_

print("\nBest parameters found:", random_search.best_params_)
print("\nBest cross-validation accuracy:", random_search.best_score_)

# Train the best model on the full training data
print("\nTraining best XGBoost model...")
best_xgb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_proba = best_xgb_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.97  # Adjust decision threshold
y_pred = (y_pred_proba >= threshold).astype(int)

# Evaluate the model
print("\nXGBoost Classification Report:\n", classification_report(y_test, y_pred))
print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
with open("best_xgboost_model.pkl", "wb") as model_file:
    pickle.dump(best_xgb_model, model_file)

print("\nModeling complete. Best XGBoost model saved as 'best_xgboost_model.pkl'.")
