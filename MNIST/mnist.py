import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
y = y.astype(np.int8)  # Convert target to integer

# Split the dataset into training and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# Use a smaller subset of the training data for the grid search
X_train, _, y_train, _ = train_test_split(X_train_full, y_train_full, train_size=5000, random_state=42)

# Perform a grid search to find the best hyperparameters for the KNeighborsClassifier
param_grid = {
    'n_neighbors': [3, 4, 5],
    'weights': ['uniform', 'distance']
}

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters found by grid search:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Train the classifier with the best parameters on the full training set
best_knn_clf = grid_search.best_estimator_
best_knn_clf.fit(X_train_full, y_train_full)

# Evaluate the classifier on the test set
y_test_pred = best_knn_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test set accuracy:", test_accuracy)
