import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('processed_data.csv', delimiter=',', header=None)

# Assume the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Load the trained Random Forest model from file
clf = joblib.load('random_forest_model.pkl')

# Predict on the test data
y_pred = clf.predict(X)

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)

# Print classification report
print("\nClasification report: \n",classification_report(y, y_pred))

# Define custom feature names
feature_names = [
    'Status of \nexisting checking \naccount', 'Duration in \nmonth', 'Credit history', 'Purpose', 'Credit amount',
    'Savings \naccount/bonds', 'Present \nemployment \nsince', 'Installment rate \nin percentage of \ndisposable income',
    'Personal \nstatus \nand sex', 'Other debtors \n/ guarantors',
    'Present residence \nsince', 'Property', 'Age in years', 'Other \ninstallment \nplans ', 'Housing',
    'Number of \nexisting credits \nat this bank', 'Job', 'Number of people \nbeing liable to provide \nmaintenance for',
    'Telephone', 'Foreign worker'
]

# Get the current working directory
current_directory = os.getcwd()

# Plot and save the first chart
plt.figure(figsize=(12, 6))
plt.bar(range(len(clf.feature_importances_[:10])), clf.feature_importances_[:10], tick_label=feature_names[:10])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances (1-10)')
plt.tight_layout()

# Save the plot as a PNG file
png_file_path_1 = os.path.join(current_directory, 'feature_importances_part1.png')
plt.savefig(png_file_path_1)

plt.close()

# Plot and save the second chart
plt.figure(figsize=(12, 6))
plt.bar(range(len(clf.feature_importances_[10:])), clf.feature_importances_[10:], tick_label=feature_names[10:])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importances (11-20)')
plt.tight_layout()

# Save the second plot as a PNG file
png_file_path_2 = os.path.join(current_directory, 'feature_importances_part2.png')
plt.savefig(png_file_path_2)

plt.close()

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix plot as a PNG file
confusion_matrix_path = os.path.join(current_directory, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)

plt.close()
