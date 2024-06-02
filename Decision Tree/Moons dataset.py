import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step a: Generate a moons dataset
X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap=plt.cm.winter)
plt.title("Moons Dataset")
plt.show()

# Step b: Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step c: Use GridSearchCV to find the best hyperparameter values
param_grid = {
    'max_leaf_nodes': [3, 4, 5, 6, 7, 10, 20, 30, 40, 50]
}
clf = DecisionTreeClassifier(random_state=42)
grid_searcher = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_searcher.fit(X_train, y_train)

print(f"Best score: {grid_searcher.best_score_}")
print(f"Best parameters: {grid_searcher.best_params_}")

# Step d: Train the model on the full training set using the best hyperparameters
best_params = grid_searcher.best_params_
clf = DecisionTreeClassifier(max_leaf_nodes=best_params['max_leaf_nodes'], random_state=42)
clf.fit(X_train, y_train)

# Measure the model's performance on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy}")

# Visualize the decision boundary
def plot_decision_boundary(clf, X, y, ax):
    x1s = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x2s = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)
    ax.contourf(x1, x2, y_pred, alpha=0.3, cmap=plt.cm.winter)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=1, cmap=plt.cm.winter)
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

fig, ax = plt.subplots(figsize=(10, 6))
plot_decision_boundary(clf, X, y, ax)
plt.title("Decision Tree Decision Boundary")
plt.show()
