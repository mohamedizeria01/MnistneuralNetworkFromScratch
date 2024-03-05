import numpy as np
import pandas as pd
import pickle
import os

# Load MNIST training dataset
data = pd.read_csv("mnist_train.csv")
data = np.array(data)
np.random.shuffle(data)

m, n = data.shape

# Transpose the dataset
data = data.T

# Separate labels and features
LABEL = data[0]
IMAGE = data[1:n]
IMAGE = IMAGE / 255.0  # Normalize pixel values

# Function to initialize or load model weights
def init_params():
    if os.path.exists('model_weights.pkl'):
        with open('model_weights.pkl', 'rb') as f:
            weights = pickle.load(f)
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
    else:
        # Initialize weights randomly
        W1 = np.random.rand(128, 784) - 0.5
        b1 = np.random.rand(128, 1) - 0.5
        W2 = np.random.rand(10, 128) - 0.5
        b2 = np.random.rand(10, 1) - 0.5
        weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        # Save initialized weights
        with open('model_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)
    return W1, b1, W2, b2

# Activation function: ReLU
def Relu(Z):
    return np.maximum(Z, 0)

# Activation function: Softmax
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation
def forward_prop(W1, b1, W2, b2, X): 
    Z1 = W1.dot(X) + b1
    A1 = Relu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# Derivative of ReLU activation function
def ReLU_deriv(Z):
    return Z > 0

# Convert labels to one-hot encoding
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

# Backward propagation
def backward_prop(Z1, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Update parameters with momentum
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, beta):
    # Initialize momentum terms
    VdW1, Vdb1, VdW2, Vdb2 = 0, 0, 0, 0
    # Update momentum terms
    VdW1 = beta * VdW1 + (1 - beta) * dW1
    Vdb1 = beta * Vdb1 + (1 - beta) * db1
    VdW2 = beta * VdW2 + (1 - beta) * dW2
    Vdb2 = beta * Vdb2 + (1 - beta) * db2
    # Update parameters with momentum
    W1 = W1 - alpha * VdW1
    b1 = b1 - alpha * Vdb1
    W2 = W2 - alpha * VdW2
    b2 = b2 - alpha * Vdb2
    return W1, b1, W2, b2

# Get predictions from output
def get_predictions(A2):
    return np.argmax(A2, 0)

# Calculate accuracy of predictions
def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# Gradient descent optimization
def gradient_descent(X, Y, alpha, iterations, beta):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha, beta)

        predictions = get_predictions(A2)
        accuracy = get_accuracy(predictions, Y)
        print("Iteration:", i, "Accuracy:", accuracy)
        if accuracy >= 0.99:
            print("Reached desired accuracy. Stopping training.")
            break
        weights = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        with open('model_weights.pkl', 'wb') as f:
            pickle.dump(weights, f)

# Train the model
gradient_descent(IMAGE, LABEL, alpha=0.5, iterations=10000, beta=0.1)
