import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the California housing dataset
housing = fetch_california_housing()

# Split the data into training, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Build the Sequential model
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),  # Hidden layer with 30 neurons and ReLU activation
    keras.layers.Dense(1)  # Output layer with 1 neuron (for regression, no activation function)
])

# Compile the model
model.compile(loss="mean_squared_error", optimizer="sgd")

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, validation_data=(X_valid_scaled, y_valid))

# Evaluate the model
mse_test = model.evaluate(X_test_scaled, y_test)
print('Test MSE:', mse_test)

# Making predictions
X_new = X_test_scaled[:3]  # pretend these are new instances
y_pred = model.predict(X_new)
print(y_pred)
