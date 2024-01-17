"""
ESN x NES implementation example.
"""
import sys
import os

from sklearn.linear_model import SGDClassifier
try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass
from pyESN import ESN
from utils import split_set, MnistDataloader, accuracy
from natural_evolution_strategies.NES import NES
import numpy as np
from sklearn.metrics import mean_squared_error


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


def test1():
    (x_train, y_train), (x_test, y_test) = load_mnist()
    esn = ESN(
        n_inputs=28*28,
        n_outputs=10,
        spectral_radius=0.8,
        n_reservoir=20,
        sparsity=0.5,
        silent=False,
        input_scaling=0.7,
        feedback_scaling=0.2,
        wash_out=25,
        learn_method="sgd",
        learning_rate=0.00001
    )
    # def f_reward(x: np.ndarray) -> float:
    #     esn_temp = ESN(**esn_args)
    #     esn_temp.W_in = x
    #     pred_train = esn.fit(x_test, y_test)
    #     return accuracy(pred_train, y_test)
    w_base = esn.W_in
    def f_reward(x: np.ndarray) -> float:
        esn.W_in = x
        pred_test = esn.predict(x_test)
        return mean_squared_error(y_test, pred_test)

    esn.fit(x_train[:5000], y_train[:5000])
    esn.silent = True

    nes = NES(
        w=esn.W_in,
        f=f_reward,
        pop=3,
        sigma=0.01,
        alpha=0.001,
    )
    nes.optimize(n_iter=20, silent=False)


if __name__ == "__main__":
    test1()

