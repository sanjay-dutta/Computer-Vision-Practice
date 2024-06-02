import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import os

# Load the Iris dataset
iris = load_iris()
X = iris.data[:, 2:]  # Using petal length and width
y = iris.target

# Train the Decision Tree Classifier
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X, y)

# Export the Decision Tree to a DOT file
os.makedirs('models/06', exist_ok=True)
export_graphviz(
    tree_clf,
    out_file='models/06/iris_tree.dot',
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)

# Convert DOT file to a PNG file and visualize it
with open("models/06/iris_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).render(filename='models/06/iris_tree', format='png', cleanup=True)

# Display the generated tree
from IPython.display import Image
Image(filename='models/06/iris_tree.png')
