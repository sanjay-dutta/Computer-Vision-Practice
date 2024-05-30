import numpy as np
import pandas as pd
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Function to shift an MNIST image
def shift_image(image, direction):
    if direction == 'left':
        return shift(image.reshape(28, 28), [0, -1], cval=0).reshape(784)
    elif direction == 'right':
        return shift(image.reshape(28, 28), [0, 1], cval=0).reshape(784)
    elif direction == 'up':
        return shift(image.reshape(28, 28), [-1, 0], cval=0).reshape(784)
    elif direction == 'down':
        return shift(image.reshape(28, 28), [1, 0], cval=0).reshape(784)

# Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist['data'], mnist['target']
y = y.astype(np.int8)  # Convert target to integer

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

# Data augmentation: shift each image in the training set
directions = ['left', 'right', 'up', 'down']
X_train_augmented = [X_train]
y_train_augmented = [y_train]

for direction in directions:
    X_shifted = np.apply_along_axis(shift_image, 1, X_train, direction)
    X_train_augmented.append(X_shifted)
    y_train_augmented.append(y_train)

# Concatenate the original and augmented datasets
X_train_augmented = np.concatenate(X_train_augmented)
y_train_augmented = np.concatenate(y_train_augmented)

# Perform a grid search to find the best hyperparameters for the KNeighborsClassifier
param_grid = {
    'n_neighbors': [3, 4, 5],
    'weights': ['uniform', 'distance']
}

knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_augmented, y_train_augmented)

print("Best parameters found by grid search:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Train the classifier with the best parameters
best_knn_clf = grid_search.best_estimator_
best_knn_clf.fit(X_train_augmented, y_train_augmented)

# Evaluate the classifier on the test set
y_test_pred = best_knn_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test set accuracy:", test_accuracy)
