import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

# Generate a linearly separable dataset
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.0)

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train a LinearSVC
linear_svc = LinearSVC(C=1, max_iter=10000)
linear_svc.fit(X_scaled, y)

# Train an SVC with a linear kernel
svc = SVC(kernel='linear', C=1)
svc.fit(X_scaled, y)

# Train an SGDClassifier with a linear kernel
sgd_clf = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
sgd_clf.fit(X_scaled, y)

# Plot the decision boundaries
def plot_decision_boundary(clf, X, y, ax):
    x0s = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    x1s = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X_new).reshape(x0.shape)
    ax.contourf(x0, x1, y_pred, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k')
    ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
    ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plot_decision_boundary(linear_svc, X_scaled, y, axs[0])
axs[0].set_title('LinearSVC')
plot_decision_boundary(svc, X_scaled, y, axs[1])
axs[1].set_title('SVC with linear kernel')
plot_decision_boundary(sgd_clf, X_scaled, y, axs[2])
axs[2].set_title('SGDClassifier')
plt.show()
