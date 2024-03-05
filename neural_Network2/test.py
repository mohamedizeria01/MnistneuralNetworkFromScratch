import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load MNIST test dataset
data = pd.read_csv("mnist_test.csv")
data = np.array(data)
np.random.shuffle(data)

m, n = data.shape

data = data.T

LABEL = data[0]

IMAGE = data[1:n]
IMAGE = IMAGE / 255.



# Function to load weights from pkl file
def load_weights(file_path):
    with open(file_path, 'rb') as f:
        weights = pickle.load(f)
    return weights

# Load the weights back
loaded_weights = load_weights('model_weights.pkl')


def Relu(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = Relu(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return A2


def get_predictions(A2):
    return np.argmax(A2, 0)


def make_predictions(X, W1, b1, W2, b2):
    A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


# Test a single prediction with a randomly chosen sample
def test_prediction(W1, b1, W2, b2):
    # Generate a random index within the range of the dataset
    index = np.random.randint(0, len(LABEL))
    
    current_image = IMAGE[:, index][:, None]
    
    prediction = make_predictions(current_image, W1, b1, W2, b2)
    
    label = LABEL[index]
    
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# Test the model with a randomly chosen sample
test_prediction(loaded_weights['W1'], loaded_weights['b1'], loaded_weights['W2'], loaded_weights['b2'])
