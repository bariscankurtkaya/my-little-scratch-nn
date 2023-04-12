import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

np.random.seed(0)

data = pd.read_csv("../../dataset/train.csv")

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)  # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape


def init_params():
    W1 = np.random.rand(10, 784) * np.sqrt(1./(784))
    b1 = np.random.rand(10, 1)
    W2 = np.random.rand(10, 10) * np.sqrt(1./(10))
    b2 = np.random.rand(10, 1)
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = Leaky_ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0

def Leaky_ReLU(Z):
    Z = np.where(Z > 0, Z, Z * 0.1)
    return Z
    
def Leaky_ReLU_deriv(Z):
    Z = np.where(Z > 0, 1, 0.1)
    return Z

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * Leaky_ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def inv_Leaky_ReLU(Z):
    Z = np.where(Z > 0, Z, Z * 10)
    return Z

def inverse_nn(Y, W1, b1, W2, b2):
    print(Y.shape)
    Z2 = Y # 10, 1
    print(W2.T.shape, Z2.shape, "+", b2.shape)

    invZ2 = Z2 - b2
    A1 = W2.T.dot(invZ2) # 10 , 1
    print(A1.shape)
    Z1 = inv_Leaky_ReLU(A1)# 10, 1
    
    invZ1 = Z1 - b1
    print(W1.T.shape, Z1.shape, "+", b1.shape)
    X = W1.T.dot(invZ1) # 784 ,1
    
    X = X.reshape((28,28)) * 255
    return X

if __name__ == "__main__":
    W1 = 0
    b1 = 0
    W2 = 0
    b2 = 0
    index = 0
    if index == 0:
        print("Train")
        W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 250)
        index = 3 #1

    if index == 1:
        print("Test")
        test_prediction(0, W1, b1, W2, b2)
        test_prediction(1, W1, b1, W2, b2)
        test_prediction(2, W1, b1, W2, b2)
        test_prediction(3, W1, b1, W2, b2)
        index = 2

    if index == 2:
        print("Accuracy")
        dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
        print(get_accuracy(dev_predictions, Y_dev))
        index = 3

    if index == 3:
        print("Inverse NN")
        const = 54.44506444837289
        Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).reshape((10,1))
        
        for i in range(10):
            Y[i] = const
            if i > 0:
                Y[i-1] = 0

            inn = inverse_nn(Y = Y, W1= W1, b1= b1, W2 = W2, b2 = b2)

            plt.gray()
            plt.imshow(inn, interpolation='nearest')
            plt.show()
