import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load the dataset
data = pd.read_csv('processed_data.csv', delimiter=',', header=None)

# Assume the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Adjust target variable to start from 0
y = y - 1

# Initialize the Gradient Boosting classifier
clf = GradientBoostingClassifier(random_state=69)

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=69)

# Lists to store the results
cv_scores = []

# Perform K-Fold cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    
    # Split the data into training and validation sets for this fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Train the model on the training set
    clf.fit(X_train, y_train)
    
    # Evaluate the model on the validation set
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores.append(accuracy)
    
    # Print the classification report and confusion matrix for this fold
    print(f"Classification Report for fold {fold + 1}:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix for fold {fold + 1}:\n{confusion_matrix(y_test, y_pred)}")

# Print cross-validation scores and their mean
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", np.mean(cv_scores))

# # Save the final model
# joblib.dump(clf, 'GBM_model.pkl')
# print("Final trained model saved")