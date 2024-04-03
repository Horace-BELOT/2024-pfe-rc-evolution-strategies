"""
This file contains scripts 
"""
import sys
import os
import multiprocessing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier
from typing import Dict, List, Any, Optional

try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass
from pyESN import ESN
from utils import split_set, MnistDataloader, accuracy, pinv, save_pickle, load_pickle
from natural_evolution_strategies.NES import NES

import umap


INPUT_PATH = 'data'
TRAINING_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
TRAINING_LABELS_FILEPATH = os.path.join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
TEST_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
TEST_LABELS_FILEPATH = os.path.join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

def load_mnist():
    """Loads the MNIST dataset"""
    mnist_dataloader = MnistDataloader(TRAINING_IMAGES_FILEPATH, TRAINING_LABELS_FILEPATH, TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(normalize=True)
    return (x_train, y_train), (x_test, y_test)

def mnist_reduced(input_dim: int = 4):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    reducer = umap.UMAP(n_components=input_dim)
    x_train = reducer.fit_transform(x_train)
    x_test = reducer.transform(x_test)
    return (x_train, y_train), (x_test, y_test)


def esn_mnist_umap_io(
        input_dim: int = 4,
        reservoir_size: int = 50,
):
    """
    Reduces dimension of MNIST input data using UMAP reducer.
    Then, trains ESN input AND output layer simultaneously using NES.

    Args:
        input_dim: int
            Dimension of input data after UMAP reduction
        reservoir_size: int
            Number of neurons in the reservoir
    """
    (x_train, y_train), (x_test, y_test) = mnist_reduced(input_dim)
    esn = ESN(
        n_inputs=input_dim,
        n_outputs=10,
        spectral_radius=0.8,
        n_reservoir=reservoir_size,
        sparsity=0.5,
        silent=False,
        input_scaling=0.7,
        feedback_scaling=0.2,
        wash_out=25,
        learn_method="pinv",
        random_state=12,
        allow_cut_connections=False
    )

    # We start from an array fitted on 100 samples
    def f_reward(w_temp: np.ndarray) -> float:
        esn.W_in = w_temp[:esn.n_inputs, :]
        esn.W_out = w_temp[esn.n_inputs:, :]
        pred_train = esn.predict(x_train, continuation=False)
        return (-np.linalg.norm(pred_train - y_train) /
                 (y_train.shape[0] * y_train.shape[1]))

    # W_in = n_reservoir * n_inputs ==> we transpose
    # W_out = n_outputs * n_reservoir
    w_nes = np.vstack([esn.W_in.T, esn.W_out]).copy()
    nes = NES(
        w=w_nes,
        f=f_reward,
        pop=25,
        sigma=5 * 10 ** (-10),
        alpha=0.01,
        mirrored_sampling=True,
    )

    nes.optimize(n_iter=50, graph=False, silent=False)

    pred_train = esn.predict(x_train, continuation=False)
    pred_test = esn.predict(x_test, continuation=False)

    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)
    print(f"Training accuracy: {100*train_acc:.2f}%")
    print(f"Testing accuracy: {100*test_acc:.2f}%")


def main():
    esn_mnist_umap_io()


if __name__ == "__main__":
    main()