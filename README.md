#  Fraud Detection Project

##  B — Background

The purpose of this project is to develop a machine learning model that detects fraudulent transactions. Fraud detection is a critical component for financial institutions to minimize financial losses and protect customers.

##  U — Understanding

The transaction dataset used for this project contains over 6 million financial transactions, providing a substantial sample size for building robust fraud detection models. Out of these, over 1.2 million transactions were used as the test set for evaluating each model.

The dataset included the following features:: step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, and newbalanceDest. 

Additional features engineered include:: sender_percentage_withdrawn, recipient_percentage_increased

Data cleaning involved handling missing values and converting transaction types to categorical data. However, due to the lack of time and date features, creating a detailed transaction history was challenging.

##  I — Implementation

### Models Tested:

Logistic Regression

XGBoost

### Evaluation Metrics:

Recall was prioritized to minimize missed fraudulent transactions.

Precision was considered to limit false positives.

F1-Score for a balance between precision and recall.

Hyperparameter tuning was conducted using RandomizedSearchCV along with cross-validation. The XGBoost model showed better overall performance by correctly predicting more fraudulent transactions while minimizing missed fraud cases. The Logistic Regression model, though less accurate overall, had fewer false positives.

##  L — Lessons Learned

Managing imbalanced data was a significant challenge; however, feature engineering boosted model performance.

The balance between recall and precision was carefully managed throughout the development of each model.

This project honed my quantitative skills, including leveraging sklearn, pandas, and numpy, and improved my abilities in model building, testing, and hyperparameter tuning.

Navigating issues like underfitting, overfitting, and selecting appropriate train-test splits was a valuable experience.

##  D — Deliverables

### Code Files:

load_data.py

clean_data.py

feature_engineering.py

data_processing.py

modeling.py

logistic regression.py

Best Models.py

XGBoost Feature Importance.py

Logistic Regression Feature Importance.py

Data Visualization.py

###  Outputs:

Confusion matrices visualizing model performance.

Feature importance visualizations for XGBoost and Logistic Regression.

Precision and recall analysis.

##  Running Instructions:

Clone the repository.

Download data set from here: [Fraud Detection Data Set](https://www.kaggle.com/datasets/ealaxi/paysim1/code)

Ensure all dependencies are installed (sklearn, pandas, numpy, matplotlib, etc.).

Run the Python files in the order listed above. 

**Note: This Project is built using Python 3.13.0**

**Note: Model training may take time depending on your system's performance. The final outputs include PNGs displaying model evaluation metrics and visualizations.**
