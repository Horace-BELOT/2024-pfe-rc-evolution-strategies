"""

"""
import os
import sys
from math import sqrt
import numpy as np
from typing import List, Dict, Tuple

try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass
import matplotlib.pyplot as plt
from natural_evolution_strategies.NES import NES
from utils import pinv



def nes_test_simple():
    """
    In this example here, we want to find a "mystery" matrix
    using NES. We will define a 1D or 2D matrix and we will use
    as reward function -distance(x, target_matrix).
    The NES algorithm converges towards the matrix.
    """
    # target_array = np.array([[  1,  4,  5, -6]])

    target_array = np.array([[  1,  4,  5,  6],
                             [  2,  3, -2, -7],
                             [ -1,  0,  1,  0]])
    base_array = np.zeros_like(target_array, dtype=float)
    # reward_func = lambda x: -sqrt(((target_array - x)**2).sum())
    reward_func = lambda x: - np.linalg.norm(target_array - x) / 12
    nes = NES(
        w=base_array,
        f=reward_func,
        pop=100,
        sigma=5 * 10 ** (-5),
        alpha=5 * 10 ** (-1)
    )
    n_iter: int = 250
    res: List[np.ndarray] = []
    alpha_array: List[float] = []
    sigma_array: List[float] = []
    for i in range(15):
        loss_array = nes.optimize(n_iter=n_iter, silent=False)
        alpha_array += [nes.alpha for _ in range(n_iter)]
        sigma_array += [nes.sigma for _ in range(n_iter)]
        res.append(loss_array)
        nes.alpha /= 10
    print(nes.w)
    plt.plot(-np.log10(-np.hstack(res)), label="loss")
    plt.plot(-np.log10(np.array(alpha_array)), label="alpha")
    plt.plot(-np.log10(np.array(sigma_array)), label="sigma")
    plt.show()


def nes_test_high_dimension(n: int, p: int, lower_bound: float = -1, upper_bound: float = 1):
    """
    This example is almost identical to the simple model: optimize the
    L2 distance between a matrix and a target matrix; Here we will draw
    a target matrix in n * p with weights drawn in U(lower_bound, upper_bound)

    """
    target_array = (upper_bound - lower_bound) * np.random.rand(n, p) + lower_bound
    base_array = np.zeros_like(target_array, dtype=float)
    reward_func = lambda x: - np.linalg.norm(target_array - x) / (n * p)
    nes = NES(
        w=base_array,
        f=reward_func,
        pop=25,
        sigma=5 * 10 ** (-1),
        alpha=5 * 10 ** (-1)
    )
    loss_array = nes.optimize(n_iter=400, silent=False, graph=True)
    print(nes.w)
    plt.plot(-np.log10(-loss_array), label="loss")
    plt.show()
    return


def nes_test_regression(

    ):
    """
    Here we have a function:
    y = AX + epsilon (white noise of std=noise)
    We want to optimize A as to minimize the distance (y - AX) 
    """
    n_samples: int = 1000
    n_targets: int = 10
    n_inputs: int = 100
    noise: float = 0.1
    a, b = (-1, 1)
    x: np.ndarray = (a - b) * np.random.rand(n_samples, n_inputs) + a
    # Initializing the hidden target matrix of coeff
    a, b = (-1, 1)
    A_target: np.ndarray = (a - b) * np.random.rand(n_targets, n_inputs) + a
    # Building y data
    # y: np.ndarray = np.zeros(shape=(n_samples, n_targets))
    y: np.ndarray = np.dot(A_target, x.T).T
    # Adding the noise
    y += np.random.normal(loc=0, scale=noise, size=(n_samples, n_targets))

    # NES part
    # initialization the matrix to optimize:
    A_opt: np.ndarray = np.zeros_like(A_target)
    def reward_func(z: np.ndarray) -> float:
        """Reward function to optimize. - L2_loss(AX - y)"""
        # return - np.sqrt(np.sum(np.square(np.dot(z, x.T).T - y)))
        return - np.linalg.norm(np.dot(z, x.T).T - y) / (n_samples * n_targets)
    
    nes = NES(
        w=A_opt,
        f=reward_func,
        pop=50,
        sigma=5 * 10 ** (-6),
        alpha=5 * 10 ** (-1),
        mirrored_sampling=False,
    )
    training_loss: np.ndarray = nes.optimize(n_iter=150, silent=False, graph=True)
    nes.alpha = 5 * 10 ** (-2)
    training_loss: np.ndarray = nes.optimize(n_iter=150, silent=False, graph=True)
    # We then want to compute the optimal value
    A_opt = pinv(x, y)
    print(f"Optimal loss: {reward_func(A_opt)}")

    return



if __name__ == "__main__":
    # nes_test_simple()
    # nes_test_high_dimension(n=100, p=10, lower_bound=-1, upper_bound=-1)
    nes_test_regression()
    pass