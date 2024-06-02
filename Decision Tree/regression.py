import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import matplotlib.pyplot as plt
import os
import graphviz

# Generate a noisy quadratic dataset
X = np.linspace(start=0, stop=1, num=500)
y = (X - 0.5) ** 2 + np.random.randn(500) / 50.
plt.scatter(X, y, s=1.5, c='red')
plt.title("Noisy Quadratic Dataset")
plt.xlabel("X")
plt.ylabel("y")
plt.show()

# Train a Decision Tree Regressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X.reshape(-1, 1), y)

# Print the trained DecisionTreeRegressor
print(tree_reg)

# Create directory for model output if it does not exist
os.makedirs('models/06', exist_ok=True)

# Export the Decision Tree to a DOT file
export_graphviz(
    tree_reg,
    out_file='models/06/reg_tree.dot',
    feature_names=['X'],
    rounded=True,
    filled=True
)

# Convert DOT file to a PNG file and visualize it
with open("models/06/reg_tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph).render(filename='models/06/reg_tree', format='png', cleanup=True)

# Display the generated tree
from IPython.display import Image
Image(filename='models/06/reg_tree.png')
