import numpy as np
import matplotlib.pyplot as plt

# Softmax function
def softmax(logits):
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

# Cross-entropy loss function
def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

# One-hot encoding
def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

# Batch Gradient Descent with early stopping and loss recording
def batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000, patience=10):
    m, n = X.shape
    num_classes = len(np.unique(y))
    y_encoded = one_hot_encode(y, num_classes)
    W = np.random.randn(n, num_classes)
    b = np.zeros((1, num_classes))
    
    best_loss = np.inf
    epochs_no_improve = 0
    loss_history = []

    for epoch in range(epochs):
        logits = np.dot(X, W) + b
        y_pred = softmax(logits)
        loss = cross_entropy(y_encoded, y_pred)
        
        loss_history.append(loss)
        print(f"Epoch {epoch}, Loss: {loss}")
        
        if loss < best_loss:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        gradient_W = np.dot(X.T, (y_pred - y_encoded)) / m
        gradient_b = np.sum(y_pred - y_encoded, axis=0, keepdims=True) / m
        
        W -= learning_rate * gradient_W
        b -= learning_rate * gradient_b
    
    return W, b, loss_history

# Example usage with random data
np.random.seed(42)
X = np.random.randn(1000, 5)
y = np.random.randint(0, 3, 1000)

W, b, loss_history = batch_gradient_descent(X, y, learning_rate=0.01, epochs=1000, patience=10)

# Save the plot
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.savefig('loss_curve.png')  # Save the plot as a file
plt.close()  # Close the plot to free up memory

print(f"Final weights: {W}")
print(f"Final biases: {b}")
