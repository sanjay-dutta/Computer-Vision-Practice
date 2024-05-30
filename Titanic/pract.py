import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the training dataset
train_df = pd.read_csv('/impacs/sad64/SLURM/dataset/titanic/train.csv')
# Load the test dataset
test_df = pd.read_csv('/impacs/sad64/SLURM/dataset/titanic/test.csv')

# Display the first few rows of the training dataset
print(train_df.head())

# Get a summary of the dataset
print(train_df.info())

# Get descriptive statistics
print(train_df.describe())

# Fill missing values for the 'Age' feature with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Fill missing values for the 'Embarked' feature with the most common port
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

# Fill missing values for the 'Fare' feature in the test set with the median fare
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Convert 'Sex' feature to numerical
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' feature to numerical
train_df['Embarked'] = train_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_df['Embarked'] = test_df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Drop irrelevant features
train_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Create a new feature 'FamilySize' from 'SibSp' and 'Parch'
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

# Create a new feature 'IsAlone'
train_df['IsAlone'] = np.where(train_df['FamilySize'] > 1, 0, 1)
test_df['IsAlone'] = np.where(test_df['FamilySize'] > 1, 0, 1)

# Prepare the data for training
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the validation set
y_val_pred = clf.predict(X_val)

# Evaluate the model on the validation set
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation set accuracy: {val_accuracy:.2f}')

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 6, 8, 10],
    'criterion': ['gini', 'entropy']
}

# Perform grid search
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best parameters found by grid search: {best_params}')

# Train the classifier with the best parameters
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)

# Predict on the validation set with the best model
y_val_pred = best_clf.predict(X_val)

# Evaluate the model on the validation set
val_accuracy_tuned = accuracy_score(y_val, y_val_pred)
print(f'Validation set accuracy after tuning: {val_accuracy_tuned:.2f}')

# Since the test set does not have labels, we cannot compute test accuracy directly.
# We can only generate predictions for the test set.

# Predict on the actual test set
test_pred = best_clf.predict(test_df.drop(['PassengerId'], axis=1))

# Print the predictions for the test set
print("Test set predictions:")
print(test_pred)

# Save the test set predictions to a text file
with open("test_set_predictions.txt", "w") as f:
    f.write("Test set predictions:\n")
    f.write(np.array2string(test_pred, separator=', '))

# Prepare the submission file
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': test_pred
})
submission.to_csv('titanic_submission.csv', index=False)

print("Submission file created: 'titanic_submission.csv'")
print("Test set predictions saved to 'test_set_predictions.txt'")
