import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Input datasets
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Seed the random number generator for reproducibility
np.random.seed(42)

# Initialize weights randomly with mean 0
weights_input_hidden = np.random.uniform(size=(2, 2))
weights_hidden_output = np.random.uniform(size=(2, 1))

# Learning rate
lr = 0.1

# Number of epochs for training
epochs = 10000

# Training the neural network
for i in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output))
    
    # Backward propagation
    # Calculate the error
    error = y - final_output
    d_predicted_output = error * sigmoid_derivative(final_output)
    
    # Calculate the error for the hidden layer
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # Update the weights
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * lr
    weights_input_hidden += X.T.dot(d_hidden_layer) * lr

# Testing the output for every input combination
print("\nOutput from neural network after {} epochs: ".format(epochs))
for x, target in zip(X, y):
    hidden_layer_input = np.dot(x, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    final_output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output))
    print("Input:{} - Predicted Output:{} - Actual Output:{}".format(x, final_output, target))
