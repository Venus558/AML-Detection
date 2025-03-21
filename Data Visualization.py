import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the confusion matrix data
conf_matrix_df = pd.read_csv('confusion_matrix_with_labels.csv')

# Load the XGBoost feature importance
xgb_importance_df = pd.read_pickle('XGBoost Feature Importance.pkl')

# Load the Logistic Regression feature importance
logreg_importance_df = pd.read_pickle('Logistic Regression Feature Importance.pkl')

# Visualizing Feature Importance (XGBoost vs Logistic Regression)
# Initialize the figure, but do not show the plots yet
fig, axes = plt.subplots(1, 2, figsize=(14, 7))  # Increase figure width for better spacing

# XGBoost feature importance
ax1 = sns.barplot(x='Importance', y='Feature', data=xgb_importance_df, palette='Blues_d', ax=axes[0])
ax1.set_title('XGBoost Feature Importance')

# Annotate with values (4 decimal points), ensuring the values stay within the chart bounds
for p in ax1.patches:
    value = p.get_width()
    x_position = value + 0.05 if value + 0.05 <= ax1.get_xlim()[1] else ax1.get_xlim()[1] - 0.05
    ax1.text(x_position, p.get_y() + p.get_height() / 2, f'{value:.4f}', ha='center', va='center', fontsize=10)

# Rotate y-tick labels for better readability
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

# Logistic Regression feature importance
ax2 = sns.barplot(x='Importance', y='Feature', data=logreg_importance_df, palette='Reds_d', ax=axes[1])
ax2.set_title('Logistic Regression Feature Importance')

# Annotate with values (4 decimal points), ensuring the values stay within the chart bounds
for p in ax2.patches:
    value = p.get_width()
    x_position = value + 0.05 if value + 0.05 <= ax2.get_xlim()[1] else ax2.get_xlim()[1] - 0.05
    ax2.text(x_position, p.get_y() + p.get_height() / 2, f'{value:.4f}', ha='center', va='center', fontsize=10)

# Adjust layout
plt.tight_layout()

# Save the Figure as a png
plt.savefig('feature_importance_comparison.png')

# Visualizing Confusion Matrix for XGBoost and Logistic Regression

# Reshape the confusion matrix DataFrame
xgb_conf_matrix = conf_matrix_df['XGBoost Confusion Matrix'].values.reshape(2, 2)
logreg_conf_matrix = conf_matrix_df['Logistic Regression Confusion Matrix'].values.reshape(2, 2)

# Set up the plot for confusion matrix comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

# XGBoost Confusion Matrix
sns.heatmap(xgb_conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
            xticklabels=['Predicted No Fraud', 'Predicted Fraud'], 
            yticklabels=['True No Fraud', 'True Fraud'], ax=axes2[0], 
            cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)

# Manually set color for each cell with improved green and red colors, plus increased contrast
for i in range(2):
    for j in range(2):
        if i == j:  # Correct predictions (True Negative or True Positive)
            if i == 0:  # True Negative (Top-left)
                color = '#006400'  # Dark green for True Negative
                alpha = 0.9
            else:  # True Positive (Bottom-right)
                color = '#00FF00'  # Bright green for True Positive
                alpha = 0.7
        else:  # Incorrect predictions (False Positive or False Negative)
            if i == 0:  # False Positive (Top-right)
                color = '#8B0000'  # Dark red for False Positive
                alpha = 0.9
            else:  # False Negative (Bottom-left)
                color = '#FF6347'  # Lighter red for False Negative
                alpha = 0.7
        axes2[0].add_patch(plt.Rectangle((j, i), 1, 1, color=color, alpha=alpha))

axes2[0].set_title('XGBoost Confusion Matrix')

# Logistic Regression Confusion Matrix
sns.heatmap(logreg_conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
            xticklabels=['Predicted No Fraud', 'Predicted Fraud'], 
            yticklabels=['True No Fraud', 'True Fraud'], ax=axes2[1], 
            cbar_kws={'shrink': 0.8}, vmin=0, vmax=1)

# Manually set color for each cell with improved green and red colors, plus increased contrast
for i in range(2):
    for j in range(2):
        if i == j:  # Correct predictions (True Negative or True Positive)
            if i == 0:  # True Negative (Top-left)
                color = '#006400'  # Dark green for True Negative
                alpha = 0.9
            else:  # True Positive (Bottom-right)
                color = '#00FF00'  # Bright green for True Positive
                alpha = 0.7
        else:  # Incorrect predictions (False Positive or False Negative)
            if i == 0:  # False Positive (Top-right)
                color = '#8B0000'  # Dark red False Positive
                alpha = 0.9
            else:  # False Negative (Bottom-left)
                color = '#FF6347'  # Lighter red for False Negative
                alpha = 0.7
        axes2[1].add_patch(plt.Rectangle((j, i), 1, 1, color=color, alpha=alpha))

axes2[1].set_title('Logistic Regression Confusion Matrix')

# Adjust layout and do not display the plot immediately
plt.tight_layout()

# Save the Figure as a png
plt.savefig('confusion_matrix_comparison.png')

# Display the Plots
plt.show()

# Add Precision and Recall Visualization Based on Confusion Matrix

plt.close('all')

# Extract values from confusion matrices
xgb_TN, xgb_FP = xgb_conf_matrix[0]
xgb_FN, xgb_TP = xgb_conf_matrix[1]
logreg_TN, logreg_FP = logreg_conf_matrix[0]
logreg_FN, logreg_TP = logreg_conf_matrix[1]

# Calculate recall and precision
xgb_recall = xgb_TP / (xgb_TP + xgb_FN) * 100 if (xgb_TP + xgb_FN) > 0 else 0
xgb_precision = xgb_TP / (xgb_TP + xgb_FP) * 100 if (xgb_TP + xgb_FP) > 0 else 0
logreg_recall = logreg_TP / (logreg_TP + logreg_FN) * 100 if (logreg_TP + logreg_FN) > 0 else 0
logreg_precision = logreg_TP / (logreg_TP + logreg_FP) * 100 if (logreg_TP + logreg_FP) > 0 else 0

# Grouped bar chart data (x = Metric group, hue = Model)
performance_df = pd.DataFrame({
    'Metric': ['Recall', 'Recall', 'Precision', 'Precision'],
    'Model': ['XGBoost', 'Logistic Regression', 'XGBoost', 'Logistic Regression'],
    'Percentage': [xgb_recall, logreg_recall, xgb_precision, logreg_precision]
})

# Custom colors for model bars (XGBoost = blue/light blue, LogReg = red/light red)
palette = {
    'XGBoost': '#1f77b4',               # Dark Blue
    'Logistic Regression': '#d62728'    # Dark Red
}

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
barplot = sns.barplot(x='Metric', y='Percentage', hue='Model', data=performance_df, palette=palette, dodge=True, ax=ax)

# Add percentage labels
for container in barplot.containers:
    ax.bar_label(container, fmt='%.1f%%', fontsize=10, padding=3)

# Final formatting
ax.set_title('Fraud Detection Performance: Recall vs Precision')
ax.set_ylabel('Percentage (%)')
ax.set_ylim(0, 110)
ax.legend(title='Model', loc='upper right')

# Adjust layout
fig.tight_layout()

# Save the Figure as a png
fig.savefig('fraud_detection_performance.png')

# Display the Final Chart
plt.show()
