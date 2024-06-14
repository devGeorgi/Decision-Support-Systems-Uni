import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('processed_data.csv', delimiter=',', header=None)

# Assume the last column is the target variable and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Initialize the Random Forest classifier
clf = RandomForestClassifier(random_state=69)

# Train the final model on the entire dataset
clf.fit(X, y)

# #Save model
# joblib.dump(clf, 'random_forest_model.pkl')

# print("Trained model saved as 'random_forest_model.pkl'")