# CS253-Assignment
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import  GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
iowa_file_path='/kaggle/input/who-is-the-real-winner/train.csv'
test_file_path='/kaggle/input/who-is-the-real-winner/test.csv'
home_data = pd.read_csv(iowa_file_path)
test_data=pd.read_csv(test_file_path)

features=['Criminal Case','Total Assets','Liabilities','state']

# Define a mapping dictionary for education levels
education_mapping = {
    'Literate': 0,
    '5th Pass': 1,
    '8th Pass': 2,
    '10th Pass': 3,
    '12th Pass': 4,
    'Graduate': 5,
    'Graduate Professional': 6,
    'Post Graduate': 7,
    'Doctorate': 8,
    'Others': 9  
}
# Map the education levels to numerical values in the 'Education' column
home_data['Education'] = home_data['Education'].map(education_mapping)

# Define a mapping dictionary for suffixes
suffix_mapping = {
    'Crore+': 1e7,   # 10 million
    'Lac+': 1e5,     # 100 thousand
    'Thou+': 1e3,    # 1 thousand
    'Hund+': 1e2     # 100
}
# Define a function to convert string representations with suffixes to numeric values
def convert_to_numeric(value):
    parts = value.split()
    number = float(parts[0])
    
    # If there's a suffix, multiply the number by its multiplier
    if len(parts) > 1 and parts[1] in suffix_mapping:
        number *= suffix_mapping[parts[1]]
    
    return number

# Apply the function to the 'Total Assets' column
home_data['Total Assets'] = home_data['Total Assets'].apply(lambda x: convert_to_numeric(x) if '+' in x else float(x))
test_data['Total Assets'] = test_data['Total Assets'].apply(lambda x: convert_to_numeric(x) if '+' in x else float(x))

# Apply the function to the 'Liabilities' column
home_data['Liabilities'] = home_data['Liabilities'].apply(lambda x: convert_to_numeric(x) if '+' in x else float(x))
test_data['Liabilities'] = test_data['Liabilities'].apply(lambda x: convert_to_numeric(x) if '+' in x else float(x))
# Perform one-hot encoding for 'state' and 'Party' columns
home_data = pd.get_dummies(home_data, columns=['state', 'Party'])
test_data = pd.get_dummies(test_data, columns=['state', 'Party'])


home_data.drop(columns=['ID', 'Candidate','Constituency ∇'], inplace=True)
test_data.drop(columns=['ID', 'Candidate','Constituency ∇'], inplace=True)


# Update the features list after one-hot encoding
features = home_data.columns.tolist()
features.remove('Education')

X = home_data[features]
y = home_data['Education']

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.4, random_state=0)
test_X = test_data[features]

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [50],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6]
}

rf_model = RandomForestClassifier(random_state=0)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1_micro')

# Fit the grid search to the data
grid_search.fit(train_X, train_y)

# Get the best parameters
best_params = grid_search.best_params_

# Initialize RandomForestClassifier with best parameters
rf_model_best = RandomForestClassifier(**best_params, random_state=0)

# Fit the model with the best parameters
rf_model_best.fit(train_X, train_y)

# Predict on the validation set
rf_val_predictions = rf_model_best.predict(val_X)

# Reverse the education mapping dictionary
reverse_education_mapping = {v: k for k, v in education_mapping.items()}

# Map the predicted labels back to their original values
predicted_labels = [reverse_education_mapping[label] for label in rf_val_predictions]

# Convert the predicted labels to a pandas Series and assign them to a new column
pred_y = pd.Series(predicted_labels, index=val_X.index)

# Print the first few rows of the dataframe to verify the changes
# print(pred_y.head())
val_y_labels = val_y.map(reverse_education_mapping)

# Assuming rf_val_predictions contains the predicted labels and val_y contains the true labels
f1 = f1_score(val_y_labels, pred_y, average='micro')
print("F1 Score:", f1)

# For test data
rf_valtest_predictions = rf_model_best.predict(test_X)

# Map the predicted labels back to their original values
predictedtest_labels = [reverse_education_mapping[label] for label in rf_valtest_predictions]
# print(test_data.ID[:5])

# output_path = 'submission.csv'
# Initialize an empty list to store the data
output_data = []

# Iterate over the indices and predicted labels
for i, label in enumerate(predictedtest_labels):
    output_data.append({'ID': i, 'Education': label})

# Create DataFrame from the list
output = pd.DataFrame(output_data)

# Save to CSV
output.to_csv('submission.csv', index=False)
F1 Score: 0.22936893203883496
