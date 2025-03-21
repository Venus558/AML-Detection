import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform, loguniform

# Load the preprocessed (scaled) data
with open("scaled_data.pkl", "rb") as file:
    X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(file)

# Define Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

# Set up hyperparameter grid for RandomizedSearchCV
param_dist = {
    'C': loguniform(1e-4, 1e4),            # Regularization strength (inverse of regularization)
    'penalty': ['l1', 'l2'],               # Type of regularization
    'solver': ['liblinear', 'saga'],       # Optimization algorithm
    'max_iter': [100, 200, 300, 400],      # Max number of iterations
    'tol': uniform(1e-5, 1e-3),            # Tolerance for stopping criteria
    'class_weight': [None, 'balanced']     # Adjust for imbalanced class weights
}

# Perform Randomized Search
random_search = RandomizedSearchCV(
    log_reg_model, param_distributions=param_dist, 
    n_iter=10, cv=2, scoring="accuracy", n_jobs=-1, verbose=2, random_state=42
)

print("Starting hyperparameter tuning...")
random_search.fit(X_train_scaled, y_train)

# Get the best model
best_log_reg_model = random_search.best_estimator_

print("\nBest parameters found:", random_search.best_params_)
print("\nBest cross-validation accuracy:", random_search.best_score_)

# Train the best model on the full training data
print("\nTraining best Logistic Regression model...")
best_log_reg_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred_proba = best_log_reg_model.predict_proba(X_test_scaled)[:, 1]
threshold = 0.5  # Default threshold for logistic regression
y_pred = (y_pred_proba >= threshold).astype(int)

# Evaluate the model
print("\nLogistic Regression Classification Report:\n", classification_report(y_test, y_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
with open("best_logistic_model.pkl", "wb") as model_file:
    pickle.dump(best_log_reg_model, model_file)

print("\nModeling complete. Best Logistic Regression model saved as 'best_logistic_model.pkl'.")
