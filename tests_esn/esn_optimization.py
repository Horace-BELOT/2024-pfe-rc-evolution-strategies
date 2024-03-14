"""
In this file will be implemented methods that will optimize the parameters
of the ESN (sparsity, spectral radius etc...) using Bayesian Optimization.

This optimization should be parallelizable since its about fitting the ESN
on MNIST data multiple times to explore the parameter space.

As an example, the grid search will also be implemented
"""
from typing import Any, Dict, List
import numpy as np
import os
import sys

try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass

from skopt import gp_minimize
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

from pyESN import ESN
from utils import MnistDataloader, accuracy

INPUT_PATH = 'data'
TRAINING_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
TRAINING_LABELS_FILEPATH = os.path.join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
TEST_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
TEST_LABELS_FILEPATH = os.path.join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')


def esn_loss_fx(x: np.ndarray, y: np.ndarray) -> float:
    """
    Args:
        x: np.ndarray
            output of the reservoir
        y: np.ndarray
            prediction on train value

    """
    return accuracy()
    
def f(params: List[Any], x: np.ndarray, y: np.ndarray, index_ref: List[int]) -> float:
    """
    
    """
    assert len(params) == 4
    spectral_radius, sparsity, input_scaling, feedback_scaling = params
    n_samples, input_size = x.shape
    esn = ESN(
        n_inputs=input_size,
        n_outputs=10,
        spectral_radius=spectral_radius,
        n_reservoir=500,
        sparsity=sparsity,
        silent=True,
        input_scaling=input_scaling,
        feedback_scaling=feedback_scaling,
        leaky_rate=0.7,
        wash_out=25,
        learn_method="pinv",
        learning_rate=0.00001,
        random_state=112,
    )
    try:
        pred = esn.fit(x, y)
        out: float = -accuracy(pred, y)
        print(f"Index: {index_ref[0]}, Accuracy: {out:.3f}, Params: {params}")
        index_ref[0] += 1
        return out
    except:
        print(f"Index: {index_ref[0]}, ERROR, Params: {params}")
        return -1


def main():
    """
    """
    # Load MINST dataset
    mnist_dataloader = MnistDataloader(TRAINING_IMAGES_FILEPATH, TRAINING_LABELS_FILEPATH, 
                                       TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
        normalize=True)

    param_grid: Dict[str, Any] = {
        "spectral_radius": np.arange(0.5,1,0.02),
        "sparsity": np.arange(0.5,1,0.02),
        "input_scaling": np.arange(0.1,1,0.1),
        "feedback_scaling": np.arange(0.1,1,0.1),
    }

    idx_ref: List[int] = [0]

    res = gp_minimize(
        func=lambda z: f(z, x_train, y_train, idx_ref),
        dimensions=[
            (0.5, 0.99),  # spectral radius
            (0.4, 0.95),  # sparsity
            (0.1, 1),  # input_scaling
            (0.1, 1),  # feedback_scaling
        ],
        acq_func="EI",
        n_calls=60,
        n_random_starts=5,
        noise=0.1**2,
        random_state=1234
    )
    print(res.x)
    plt.plot([-k for k in res.func_vals], label="Loss")
    plt.xlabel("Index")
    plt.ylabel("Accuracy")
    plt.show()

if __name__ == "__main__":
    main()