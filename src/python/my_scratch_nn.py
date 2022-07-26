import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Scratch_NN:
    def __init__(self):
        self.data_dir = "../../dataset/train.csv"

        self.init_data(self.data_dir)

        self.math_eq = Math_Equations()

       
        
    def init_data(self, direction):
        data = np.array(pd.read_csv(direction))
        self.data_count, self.class_plus_pixels_count  = data.shape
        np.random.shuffle(data)

        self.test_data = data[0:1000].T
        self.test_classes = test_data[0]
        self.test_images = test_data[1:self.class_plus_pixels_count]

        self.train_data = data[1001:self.data_count].T
        self.train_classes = train_data[0]
        self.train_images = train_data[1: self.class_plus_pixels_count]

        #To prevent the gradient from vanishing
        self.test_images = self.test_images / 255
        self.train_images = self.train_images / 255

        self.pixels_count = self.class_plus_pixels_count - 1

    def init_params(self):
        self.W1 = np.random.rand(10,784) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5


class Math_Equations:
    def __init__(self):
        continue

    def ReLU(Z):
        return np.maximum(Z, 0)

    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def ReLU_deriv(Z):
        return Z > 0


if __name__ == "__main__":
    scratch__nn = Scratch_NN()
