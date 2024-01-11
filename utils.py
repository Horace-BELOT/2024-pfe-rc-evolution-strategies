"""
This file contains essentially helper functions. Some of these are used by pyESN.
It also contains a MnistDataloader that should ease the loading of MNIST data
"""

import os
import inspect
import numpy as np
import struct
from array import array
from os.path import join
import matplotlib.pyplot as plt
import random



def load_parent_dir():
	currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
	parentdir = os.path.dirname(currentdir)
	return parentdir

def correct_dimensions(s, targetlength):
    """checks the dimensionality of some numeric arguments, broadcasts it to the specified length if possible.
    Args:
        s: None, scalar or 1D array
        targetlength: expected length of s
    Returns:
        None if s is None, else numpy vector of length targetlength
    """
    if s is not None:
        s = np.array(s)
        if s.ndim == 0:
            s = np.array([s] * targetlength)
        elif s.ndim == 1:
            if not len(s) == targetlength:
                raise ValueError("arg must have length " + str(targetlength))
        else:
            raise ValueError("Invalid argument")
    return s

def identity(x): return x

def one_hot(y, n_labels, dtype=int):
        """Returns a matrix where each sample in y is represented as a row, 
            and each column represents the class label in the one-hot encoding scheme.
        Args:
            y (1d-array): the data to be encoded
            n_labels (int): the number of categories
        """
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)    

def quantize(data, num_bins):
    bins = np.linspace(start=np.min(data), stop=np.max(data), num=num_bins, dtype=float)
    quantized = np.digitize(data, bins, right=True).astype(float)
    quantized *= (np.max(data) - np.min(data)) / (np.max(quantized) - np.min(quantized))   # scale the quantized data into the same size of the original data
    return quantized + np.min(data)  # add bias to the quantized data 


# %% Training Functions

def ridge(X, Y, ridge_noise):
    '''Compute readout weights matrix [Ridge Regression]
    Args:
        X : reservoir states matrix, (num_samples * N)
        Y : true outputs, (num_samples * output_dim)
        ridge_noise: the regularization parameter of ridge regression
    Returns:
        W_out: readout weights matrix, (output_dim * N)
    '''
    return np.dot(np.dot(Y.T, X), np.linalg.inv(np.dot(X.T, X) + ridge_noise*np.eye(X.shape[1])))


def pinv(X, Y):
    '''Compute readout weights matrix [Moore-Penrose Pseudo Inverse]
    Args & Returns: Same as self._ridge()
    '''
    return np.dot(np.linalg.pinv(X), Y).T


def split_set(x: np.ndarray, y: np.ndarray,) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    """

class MnistDataloader(object):
    """
    Source:
    https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
    """
    def __init__(
            self, 
            training_images_filepath: str,
            training_labels_filepath: str,
            test_images_filepath: str, 
            test_labels_filepath: str):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def __read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self) -> tuple[list[list[np.ndarray]], array, list[list[np.ndarray]], array]:
        x_train, y_train = self.__read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.__read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
    def prepare_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads data and returns it as lists of (1, 28x28) arrays representing image and an array representing the labels"""
        (x_train, y_train),(x_test, y_test) = self.load_data()

        x_train_numpy: np.ndarray = np.array([np.array(k).reshape((1, 28 * 28)) for k in x_train]).reshape(len(x_train), 28*28)
        y_train_numpy: np.ndarray = np.eye(10)[np.array(y_train)]  # labels
        x_test_numpy: np.ndarray = np.array([np.array(k).reshape((1, 28 * 28)) for k in x_test]).reshape(len(x_test), 28*28)
        y_test_numpy: np.ndarray = np.eye(10)[np.array(y_test)]  # labels
        return (x_train_numpy, y_train_numpy), (x_test_numpy, y_test_numpy)
    
    def show_images_sample(self):
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
        images_2_show = []
        titles_2_show = []
        for _ in range(0, 10):
            r = random.randint(1, 60000)
            images_2_show.append(x_train[r])
            titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

        for _ in range(0, 5):
            r = random.randint(1, 10000)
            images_2_show.append(x_test[r])        
            titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

        self.__show_images(images_2_show, titles_2_show)


    def __show_images(images, title_texts):
        cols = 5
        rows = int(len(images)/cols) + 1
        plt.figure(figsize=(30,20))
        index = 1    
        for x in zip(images, title_texts):        
            image = x[0]        
            title_text = x[1]
            plt.subplot(rows, cols, index)        
            plt.imshow(image, cmap=plt.cm.gray)
            if (title_text != ''):
                plt.title(title_text, fontsize = 15);        
            index += 1
        plt.show()

if __name__ == "__main__":
    # Set file paths based on added MNIST Datasets
    input_path = 'data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
