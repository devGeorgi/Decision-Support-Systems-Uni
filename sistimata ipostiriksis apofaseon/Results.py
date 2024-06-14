import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('processed_data.csv', delimiter=',', header=None)

# Assume the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Load the trained Random Forest model from file
clf = joblib.load('random_forest_model.pkl')

# # Load the trained GBM model from file
# clf = joblib.load('GBM_model.pkl')

# # Load the trained SVM model from file
# clf = joblib.load('SVM_model.pkl')

# # Load the trained XGB model from file
# clf = joblib.load('XGB_model.pkl')

# Make predictions
y_pred = clf.predict(X)

# # Make it comment if using Random Forest
# y_pred = y_pred + 1

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)

# Print confusion matrix
print(f"Confusion Matrix :\n{cm}")

# Print classification report
print("\nClassification report : \n",classification_report(y, y_pred))

# Visualize the confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
