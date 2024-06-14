import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('german.data', delimiter=' ', header=None)

# Encode categorical variables using label encoding
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':  # Check if the column is categorical
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Save the processed data to a CSV file
data.to_csv('processed_data.csv', index=False)

print('Processed data saved to processed_data.csv')
