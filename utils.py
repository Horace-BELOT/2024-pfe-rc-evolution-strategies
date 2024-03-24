"""
This file contains essentially helper functions. Some of these are used by pyESN.
It also contains a MnistDataloader that should ease the loading of MNIST data
"""

import os
import inspect
import pickle
import numpy as np
import struct
from array import array
from os.path import join
import matplotlib.pyplot as plt
import random
import tqdm
from typing import Optional, Literal, List, Tuple, Dict, Any
from skimage.feature import hog



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


def pinv(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    '''Compute readout weights matrix [Moore-Penrose Pseudo Inverse]
    Args & Returns: Same as self._ridge()
    '''
    return np.dot(np.linalg.pinv(X), Y).T


def sgd(
        X: np.ndarray, 
        Y: np.ndarray,
        alpha: float, 
        W_init: Optional[np.ndarray] = None,
        lambda_ridge: float = 0,
        silent: bool = True
        ) -> np.ndarray:
    """
    Computes simple Stochastic Gradient Descent at learning rate alpha
    Ax - y minmized
    Args:
        X: np.ndarray, inputs with shape (N, n_inputs)
        Y: np.ndarray, targets with shape (N, n_outputs)
        alpha: float, learning rate
        W_init: weight matrix starting point
        lambda_ridge: float
    Returns:
        np.ndarray, fitted weights matrix A such that AX - Y is minimized 
    """
    n_samples: int = X.shape[0]
    n_inputs: int = X.shape[1]
    n_outputs: int = Y.shape[1]
    A: np.ndarray = np.zeros(shape=(n_outputs, n_inputs))
    if W_init is not None:
        A = W_init.copy()
    
    for i in range(n_samples):
        x = X[i, :].reshape(1,-1)
        pred = np.dot(x, A.T)
        loss = pred - Y[i, :]
        if not silent and i % 100 == 0:
            cost = np.sum(loss ** 2)
            # print("Iteration %d | Cost: %f" % (i, cost))
        gradient = np.dot(loss.T, x)
        ridge_term = A * lambda_ridge
        A -= alpha * (gradient + ridge_term)
    return A


def split_set(x: np.ndarray, n: int) -> List[np.ndarray]:
    """
    Splits an array in n parts that are almost all equals. 
    (Only the last split will include the remainder and might be larger).
    The number of COLUMNS of the input array stays constant
    
    Args:
        x: np.ndarray
            Input array to split
        n: int
            Number of splits
    
    Returns
        List[np.ndarray]
            List of arrays resulting from the split
    """
    rows_per_splits: int = x.shape[0] // n
    out: List[np.ndarray] = []
    for k in range(n - 1):
        out.append(x[k * rows_per_splits:(k + 1) * rows_per_splits, :])
    out.append(x[(n - 1) * rows_per_splits:, :])
    return out


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    n, _ = y_pred.shape
    m, _ = y_true.shape
    if n != m:
        raise ValueError(f"Predicted data and True data dont have the same dimension: {n} != {m}")
    labels_pred: np.ndarray = np.array([np.argmax(y_pred[k]) for k in range(n)])
    labels_true: np.ndarray = np.array([np.argmax(y_true[k]) for k in range(n)])
    acc: float = np.sum(labels_pred == labels_true) / n
    return acc


def calc_hog_features(X, image_shape=(28, 28), 
                      cell=(8, 8),
                      block=(2,2),
                      keep_inputs: Optional[bool] = False,
                      silent: bool = True):
    """
    Converts input data into Histogram of Oriented Gradients
    """
    fd_list = []
    for row in tqdm.tqdm(X, "Computing HOG", disable=silent):
        img = row.reshape(image_shape)
        fd = hog(img, orientations=8, pixels_per_cell=cell, cells_per_block=block)
        fd_list.append(fd)
    if keep_inputs:
        return np.hstack([X, np.array(fd_list)])
    return np.array(fd_list)

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
        """
        Initializes the object with the path of the data 
        """
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
        input_to_output_ratio: int = 1
    
    def __read_images_labels(self, images_filepath, labels_filepath):        
        labels =     []
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
            
    def load_data(self) -> Tuple[List[List[np.ndarray]], array, List[List[np.ndarray]], array]:
        x_train, y_train = self.__read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.__read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    
    def prepare_data(
            self,
            normalize: bool = False,
            crop_top: int = 0,
            crop_bot: int = 0,
            crop_left: int = 0,
            crop_right: int = 0,
            out_format: Literal["normal", "column", "row"] = "normal",
            projection: Optional[int] = None,
            hog: Optional[Dict[Literal["image_shape", "cell", "block", ], Any]] = None,
            silent: bool = True
        ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Loads MNIST data arrays representing image & the labels
        The transformation can be (in order):
            - Cropping
            - HOG
            - Projecting
            - Normalizing

        Inputs:
            normalize: bool
                normalize inputs
            projection: Optional[int]
                projects randomly the data on a lower dimension space.
            hog: Optional[Dict[Literal["image_shape", "cell", "block"], Any]]
                if not None, transforms data to HOG
                example of input can be: {"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": True}
                keep_inputs arguments keeps both the input and the HOG for each input

        """
        (x_train, y_train),(x_test, y_test) = self.load_data()

        x_train_numpy: np.ndarray = np.array([np.array(k) for k in x_train])
        y_train_numpy: np.ndarray = np.eye(10)[np.array(y_train)]
        x_test_numpy: np.ndarray = np.array([np.array(k) for k in x_test])
        y_test_numpy: np.ndarray = np.eye(10)[np.array(y_test)]

        # Changing data format
        f = lambda x, y: self.__change_array_shape(
            x, y, crop_top=crop_top, crop_bot=crop_bot, 
            crop_left=crop_left, crop_right=crop_right,
            out_format=out_format)
        x_train_numpy, y_train_numpy = f(x_train_numpy, y_train_numpy)
        x_test_numpy, y_test_numpy = f(x_test_numpy, y_test_numpy)

        # HOG
        if hog is not None:
            x_test_numpy = calc_hog_features(x_test_numpy, **hog, silent=silent)
            x_train_numpy = calc_hog_features(x_train_numpy, **hog, silent=silent)

        # Random projection
        if projection is not None:
            # We create a random projection to the given dimension
            n: int = x_train_numpy.shape[1]
            W_proj: np.ndarray = np.random.random(size=(n, projection)) * 2 - 1
            x_test_numpy = np.dot(x_test_numpy, W_proj)
            x_train_numpy = np.dot(x_train_numpy, W_proj)

        # Normalization of data
        if normalize:
            # Normalizing trainset
            m: np.ndarray = x_train_numpy.mean(axis=0)
            s: np.ndarray = x_train_numpy.std(axis=0)
            s[s == 0] = 1  # Preventing the std from being = 0
            x_train_numpy = (x_train_numpy - m) / s
            # Normalizing testset with trainset std and mean
            x_test_numpy = (x_test_numpy - m) / s
        if not silent:
            print(f"Shape of MNIST input data: {x_train_numpy.shape}")
        return (x_train_numpy, y_train_numpy), (x_test_numpy, y_test_numpy)
    
    def __change_array_shape(
            self,
            x: np.ndarray, 
            y: np.ndarray,
            crop_top: int = 0,
            crop_bot: int = 0,
            crop_left: int = 0,
            crop_right: int = 0,
            out_format: Literal["normal", "column", "row"] = "normal",
        ) -> np.ndarray:
        """
        Input:
            x: np.ndarray of shape (n_sample, 28, 28)

        Returns
            Tuple[np.ndarray, np.ndarray] of x and y arrays. x array is of shape n x p and
            y array is of shape n x 10
        """
        x_out = x[:, crop_top:, :]
        if crop_bot != 0: x_out = x_out[:, :-crop_bot, :]
        x_out = x_out[:, :, crop_left:]
        if crop_right != 0: x_out = x_out[:, :, :-crop_right]
        assert x.ndim == 3, "x array should have 3 dimension but has {x.ndim}"
        n, p, q = x_out.shape  # n samples, p rows, q columns
        y_out = y.copy()
        if out_format == "column":
            x_out = np.transpose(x_out, axes=[0, 2, 1]).reshape(n * q, p)
            y_out = np.repeat(y_out, repeats=q, axis=0)
            self.input_to_output_ratio = q
        elif out_format == "row":
            x_out = np.reshape(n * p, q)
            y_out = np.repeat(y_out, repeats=p, axis=0)
            self.input_to_output_ratio = p
        else:
            # Otherwise, 1 row = an entire image with p * q pixels
            x_out = x_out.reshape(n, p * q)
            self.input_to_output_ratio = 1
        return x_out, y_out
    
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


def test_sgd():
    from sklearn.linear_model import SGDRegressor
    n = 10000
    # X = np.random.rand(n, 3)
    # Y = (2* X[:, 0] - X[:, 1] + 0.1 * X[:, 2]).reshape(n, 1) + 0.04 * np.random.rand(n, 1) - 0.02

    X = np.random.rand(n, 3)
    Y = (0.5 * X.sum(axis=1) + 0.04 * np.random.rand(n) - 0.02).reshape(n, 1)
    Y = np.hstack([Y, 0.2 * Y])
    # clf = SGDRegressor(penalty="l2", alpha=0.0001)
    # clf.fit(X,Y)
    # clf.coef_
    A = sgd(X, Y, alpha=0.001, silent=False) 
    return


def save_pickle(obj: Dict[Any, Any], file_path: str) -> None:
    """
    Saves the content of the ESN object to a pickle fle
    """
    if ".pickle" not in file_path:
        file_path = file_path + ".pickle"
    with open(file_path, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_path: str) -> Dict[Any, Any]:
    """
    Loads an ESN object from 
    """
    with open(file_path, "rb") as handle:
        obj: Dict[str, Any] = pickle.load(handle)
    return obj

if __name__ == "__main__":
    test_sgd()
    # Set file paths based on added MNIST Datasets
    input_path = 'data'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    # Load MINST dataset
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
