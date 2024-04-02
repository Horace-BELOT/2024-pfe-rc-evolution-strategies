import gymnasium as gym
import numpy as np

import os
import sys
try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass

from utils import split_set, MnistDataloader, accuracy, pinv, save_pickle, load_pickle

import umap

def mnist_reduced(input_dim=4):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    reducer = umap.UMAP(n_components=input_dim)
    x_train = reducer.fit_transform(x_train)
    x_test = reducer.transform(x_test)
    # print max value
    print(np.max(x_train))
    return (x_train, y_train), (x_test, y_test)

# Create the MNIST environment

def load_mnist():
    """Loads the MNIST dataset"""
    input_path = 'data'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(normalize=True)
    return (x_train, y_train), (x_test, y_test)

# Continuous Action Environment
class Environnement(gym.Env):
    def __init__(self, input_dim=4, reservoir_size=50):
        super(Environnement, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(reservoir_size,))
        self.observation_space = gym.spaces.Box(low=-11, high=11, shape=(input_dim,))
        print(self.observation_space.shape)
        self.input_dim = input_dim
        if input_dim == 784:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = load_mnist()
        else:
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist_reduced(input_dim)
        self._max_episode_steps = len(self.x_train)
        self.current_step = 0
    
    def reset(self):
        self.current_step = 0
        return self.x_train[self.current_step]
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.x_train)
        if self.current_step >= len(self.x_train):
            self.current_step = 0
        reward = np.argmax(action) == np.argmax(self.y_train[self.current_step])
        return self.x_train[self.current_step], reward, done, {}
    
    def render(self):
        pass
    
    def close(self):
        pass
    
    
    
        