import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Scratch_NN:
    def __init__(self):
        self.data_dir = "../../dataset/train.csv"

        self.init_data(self.data_dir)

        self.math_eq = Math_Equations()
        self.encode = Encoding()

    
    # m = data_count
    # n = pixels_count + class
    # X_train = train_images
    # X_test = test_images
    # Y_train = train_classes
    # Y_test = test_classes
    # m_train = train_image_count

    def init_data(self, direction):
        data = np.array(pd.read_csv(direction))
        self.data_count, self.class_plus_pixels_count  = data.shape
        np.random.shuffle(data)

        self.test_data = data[0:1000].T
        self.test_classes = self.test_data[0]
        self.test_images = self.test_data[1:self.class_plus_pixels_count]

        self.train_data = data[1001:self.data_count].T
        self.train_classes = self.train_data[0]
        self.train_images = self.train_data[1: self.class_plus_pixels_count]

        #To prevent the gradient from vanishing
        self.test_images = self.test_images / 255.
        self.train_images = self.train_images / 255.

        self.pixels_count, self.train_image_count = self.train_images.shape

    def init_params(self):
        self.W1 = np.random.normal(size=(10, 784)) * np.sqrt(1./(784))
        self.b1 = np.random.normal(size=(10, 1))
        self.W2 = np.random.normal(size=(10, 10)) * np.sqrt(1./10)
        self.b2 = np.random.normal(size=(10, 1))

        self.alpha = 0.1
        self.iterations = 500

    def forward_prop(self, images):
        self.Z1 = self.W1.dot(images) + self.b1
        self.A1 = self.math_eq.Leaky_ReLU(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.math_eq.softmax(self.Z2)

    def backward_prop(self, images, classes):
        self.one_hot_classes = self.encode.one_hot(classes)
        self.dZ2 = self.A2 - self.one_hot_classes
        self.dW2 = (1/self.train_image_count) * self.dZ2.dot(self.A1.T)
        self.db2 = (1/self.train_image_count) * np.sum(self.dZ2)
        self.dZ1 = self.W2.T.dot(self.dZ2) * self.math_eq.Leaky_ReLU_deriv(self.Z1)
        self.dW1 = (1/self.train_image_count) * self.dZ1.dot(images.T)
        self.db1 = (1/self.train_image_count) * np.sum(self.dZ1)

    def update_params(self):
        self.W1 = self.W1 - self.alpha * self.dW1
        self.b1 = self.b1 - self.alpha * self.db1
        self.W2 = self.W2 - self.alpha * self.dW2
        self.b2 = self.b2 - self.alpha * self.db2

    def get_predictions(self, A2):
        return np.argmax(A2, 0)


    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size

    def gradient_descent(self):
        self.init_params()
        #print("1 ",self.W1[0][320])
        for i in range(self.iterations):
            #print("2 ",self.W1[0][320])
            self.forward_prop(self.train_images)
            #print("3 ",self.W1[0][320])
            self.backward_prop(self.train_images, self.train_classes)
            #print("4 ",self.W1[0][320])
            self.update_params()
            #print("7 ",self.W1[0][320], "\n")
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = self.get_predictions(self.A2)
                print(self.get_accuracy(predictions, self.train_classes))

    def make_predictions(self, X):
        self.forward_prop(X)
        predictions = self.get_predictions(self.A2)
        return predictions


    def test_prediction(self, index):
        current_image = self.test_images[:, index, None]
        prediction = self.make_predictions(current_image)
        label = self.train_classes[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()

class Math_Equations:
    def __init__(self) -> None:
        pass

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0 

    def Leaky_ReLU(self, Z):
        Z = np.where(Z > 0, Z, Z * 0.1)
        return Z
    
    def Leaky_ReLU_deriv(self, Z):
        Z = np.where(Z > 0, 1, 0.1)
        return Z

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

class Encoding:
    def __init__(self) -> None:
        pass
    
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

if __name__ == "__main__":
    scratch__nn = Scratch_NN()

    index = 0
    if index == 0:
        print("Train")
        scratch__nn.gradient_descent()
        #index = 1

    if index == 1:
        print("Test")
        scratch__nn.test_prediction(0)
        scratch__nn.test_prediction(1)
        scratch__nn.test_prediction(2)
        scratch__nn.test_prediction(3)
        index = 2

    if index == 2:
        print("Accuracy")
        dev_predictions = scratch__nn.make_predictions(scratch__nn.test_images)
        print(scratch__nn.get_accuracy(dev_predictions, scratch__nn.test_classes))

