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

def create_reduced(input_dim: int = 4):
    file_path = f"umap_{input_dim}.pickle"
    total_path = f"saved_data/umap/{file_path}"
    if file_path in os.listdir("saved_data/umap"):
        print(f"MNIST UMAP Data found at {total_path}. Loading")
        data: Dict[str, Any] = load_pickle(file_path=total_path)
    else:
        print(f"MNIST UMAP Data not found, recreating.")
        (x_train, y_train), (x_test, y_test) = mnist_reduced(input_dim)
        data: Dict[str, Any] = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test
        }
        save_pickle(data, total_path)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]
    return (x_train, y_train), (x_test, y_test)



def esn_mnist_umap_io(
        load_path_esn: str,
        load_path_nes: Optional[str],
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
    
    (x_train, y_train), (x_test, y_test) = create_reduced(input_dim)
    print(f"Data is ready")
    
    try: 
        esn = ESN.load(load_path_esn)
        print(f"ESN data found at {load_path_esn}. Loading")
    except:
        esn = ESN(
            n_inputs=input_dim,
            n_outputs=10,
            spectral_radius=0.8,
            n_reservoir=reservoir_size,
            sparsity=0.5,
            silent=True,
            input_scaling=0.7,
            feedback_scaling=0.2,
            wash_out=25,
            learn_method="pinv",
            random_state=12,
            allow_cut_connections=False
        )
        esn.save(load_path_esn)


    # We start from an array fitted on 100 samples
    def f_reward(w_temp: np.ndarray, saving: bool = False) -> float:
        esn.W_in = w_temp[:esn.n_inputs, :].T
        esn.W_out = w_temp[esn.n_inputs:, :]
        pred_train = esn.predict(x_train, continuation=False)
        loss = (-np.linalg.norm(pred_train - y_train) /
                 (y_train.shape[0] * y_train.shape[1]))
        if saving:
            return loss, {"train_accuracy": accuracy(pred_train, y_train)}
        return loss

    # W_in = n_reservoir * n_inputs ==> we transpose
    # W_out = n_outputs * n_reservoir
    
    w_nes = np.vstack([esn.W_in.T, esn.W_out]).copy()

    nes = NES(
        w=w_nes,
        f=f_reward,
        pop=6,
        sigma=5 * 10 ** (-10),
        alpha=0.01,
        mirrored_sampling=True,
    )
    # We try and see if there has already been a running experiment that we can continue
    try:
        nes_data = load_pickle(load_path_nes)
        print(f"Found NES data at {load_path_nes}. Loading")
        NES.w = nes_data["w"]
        nes.data = nes_data["data"]
        nes.alpha = nes.data["alpha"].iloc[-1]
        nes.sigma = nes.data["noise"].iloc[-1]
    except:
        print(f"Didn't find any NES data")

    nes.optimize(n_iter=200, graph=False, silent=False, save_path=load_path_nes)

    pred_train = esn.predict(x_train, continuation=False)
    pred_test = esn.predict(x_test, continuation=False)

    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)
    print(f"Training accuracy: {100*train_acc:.2f}%")
    print(f"Testing accuracy: {100*test_acc:.2f}%")
    plot_result()



def esn_mnist_umap_output_only(
        input_dim: int = 4,
        reservoir_size: int = 50,

):
    """
    Trains only output layer with NES
    """
    (x_train, y_train), (x_test, y_test) = create_reduced(input_dim)
    print(f"Data is ready")

    esn = ESN(
        n_inputs=input_dim,
        n_outputs=10,
        spectral_radius=0.8,
        n_reservoir=reservoir_size,
        sparsity=0.5,
        silent=True,
        input_scaling=0.7,
        feedback_scaling=0,
        wash_out=25,
        learn_method="pinv",
        random_state=12,
        allow_cut_connections=False
    )

    w_opti = esn.W_out.copy()
    nes = NES(
        w=w_opti,
        f=lambda x: None,
        pop=25,
        sigma=5 * 10 ** (-10),
        alpha=0.01,
        mirrored_sampling=True
    )
    save_path_nes_data = "saved_data/nes_mnist_output_only.csv"

    def custom_method(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """"""
        # We start from an array fitted on 100 samples
        
        def f_reward(w_temp: np.ndarray, saving: bool = False) -> float:
            pred = np.dot(x, w_temp.T)
            if saving:
                return (
                    -np.linalg.norm(pred - y) / (y.shape[0] * y.shape[1]),
                    {"train_accuracy": accuracy(pred, y)}
                )
            return -np.linalg.norm(pred - y) / (y.shape[0] * y.shape[1])
        nes.w = w_opti
        nes.f = f_reward
        nes.optimize(n_iter=1000, graph=False, save_path="saved_data/nes_mnist_output_only.csv")
        
        return w_opti
    esn.learn_method = "custom"
    esn.custom_method = custom_method
    esn.fit(x_train, y_train)


    pred_train = esn.predict(x_train, continuation=False)
    pred_test = esn.predict(x_test, continuation=False)

    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)
    print(f"Training accuracy: {100*train_acc:.2f}%")
    print(f"Testing accuracy: {100*test_acc:.2f}%")

    df = nes.data
    fig, ax1 = plt.subplots(1)
    # Training loss
    # ax1.plot(df.index, -df["train_loss"])
    # ax1.set_yscale("log")

    # Accuracy
    ax1.plot(df.index, df["train_accuracy"])
    ax1.set_xlabel("Generation")

    plt.tight_layout()
    plt.grid(True)
    plt.savefig("saved_data/nes_mnist_ouput_only_4.png")
    plt.show()

def baseline_pinv(
    input_dim: int = 4,
    reservoir_size: int = 50,
):
    (x_train, y_train), (x_test, y_test) = create_reduced(input_dim)
    print(f"Data is ready")

    esn = ESN(
        n_inputs=input_dim,
        n_outputs=10,
        spectral_radius=0.8,
        n_reservoir=reservoir_size,
        sparsity=0.5,
        silent=True,
        input_scaling=0.7,
        feedback_scaling=0,
        wash_out=25,
        learn_method="pinv",
        random_state=12,
        allow_cut_connections=False
    )
    esn.fit(x_train, y_train)

    pred_train = esn.predict(x_train, continuation=False)
    pred_test = esn.predict(x_test, continuation=False)

    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)
    print(f"Training accuracy: {100*train_acc:.2f}%")
    print(f"Testing accuracy: {100*test_acc:.2f}%")

    

def plot_result():
    load_path_nes = "saved_data/nes_mnist_io_nes_data.pickle"
    nes_data = load_pickle(load_path_nes)
    df = nes_data["data"]
    fig, ax1 = plt.subplots(1)
    # Training loss
    # ax1.plot(df.index, -df["train_loss"])
    # ax1.set_yscale("log")

    # Accuracy
    ax1.plot(df.index, df["train_accuracy"])
    ax1.set_xlabel("Generation")

    plt.tight_layout()
    plt.grid(True)
    plt.savefig("saved_data/nes_mnist_io_4.png")
    plt.show()


def main():
    # baseline_pinv()
    # plot_result()
    # esn_mnist_umap_io(load_path_esn="saved_data/nes_mnist_io_esn_data.pickle",
    #                   load_path_nes="saved_data/nes_mnist_io_nes_data.pickle")
    esn_mnist_umap_output_only()


if __name__ == "__main__":
    main()