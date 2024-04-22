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


def example_figure_pptx_nes():
    """"""
    x_arr: np.ndarray = np.linspace(-1, 1)
    y_arr: np.ndarray = x_arr ** 2 + 1
    plt.plot(x_arr, y_arr)
    plt.savefig("figures/pptx_nes_example.png")


def plot_results():
    load_path_nes = "saved_data/esn_mnist_umap_io/nes.pickle"
    nes_data = load_pickle(load_path_nes)

    df = nes_data["data"]

    plt.plot(df.index, df["train_accuracy"])
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.grid(True)
    plt.savefig("figures/nes_mnist_io_10.png")
    plt.show()

if __name__ == "__main__":
    # plot_results()
    example_figure_pptx_nes()