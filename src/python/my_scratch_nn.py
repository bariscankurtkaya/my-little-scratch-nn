import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class Scratch_NN:
    def __init__(self):
        self.data_dir = "../../dataset/train.csv"

        self.init_data(self.data_dir)

       
        
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
        


if __name__ == "__main__":
    scratch__nn = Scratch_NN()
    
