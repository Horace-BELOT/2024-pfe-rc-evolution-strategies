"""
Implementation of the NES (Natural Evolution Strategy)
"""
import numpy as np
from typing import Optional, Callable, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt


class NES:

    def __init__(
            self,
            w: np.ndarray, # base matrix
            f: Callable[[np.ndarray], float], # reward function
            pop: int = 50, # population size
            sigma: float = 0.1, # noise standard deviation
            alpha: float = 0.001, # learning rate
        ):
        """"""
        self.pop: int = pop
        self.sigma: float = sigma
        self.alpha: float = alpha
        self.w: np.ndarray = w
        self.f = f
        
    def step(self):
        """
        Computes a step of the NES.
        
        Returns:
            float
                reward function applied to the matrix AFTER the step was carried
        """
        new_gen: list[np.ndarray] = [np.random.normal(0, 1, self.w.shape) for _ in range(self.pop)]

        rewards: np.ndarray = np.zeros(self.pop)
        for i, elem in enumerate(new_gen):
            # We compute the reward for our base point + sigma * delta[i]
            rewards[i] = self.f(self.w + self.sigma * elem)
        
        # Standardize the rewards to have a gaussian distribution
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        rewards *= self.alpha / (self.pop * self.sigma)
        # Move the estimation according to the vector weighted by their rewards
        if len(self.w.shape) == 1:
            # clean way to do the computation if we have a 1D vector in input
            new_gen_matrix = np.vstack(new_gen).T # shape = (self.dim, self.pop)
            self.w = self.w + np.dot(new_gen_matrix, rewards)
        elif len(self.w.shape) == 2:
            n, p = self.w.shape
            for i in range(n):
                for j in range(p):
                    value = np.dot(
                        np.array([new_gen[k][i,j] for k in range(self.pop)]).T,
                        rewards
                    )
                    self.w[i,j] = self.w[i,j] + value
        else:
            raise ValueError(f"Weird matrix shape: {self.w.shape}")
        return self.f(self.w)
            

    def optimize(self, n_iter: int = 100, silent: bool = False) -> np.ndarray[float]:
        """
        Runs n_iter steps of the NES algorithm

        Args:
            n_iter: int
                Number of iterations
            silent: bool
                Whether to print results or not
        """
        results = np.zeros(n_iter)
        for i in tqdm(range(n_iter), disable=silent):
            results[i] = self.step()
        return results
    

if __name__ == "__main__":
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
    reward_func = lambda x: -sqrt(((target_array - x)**2).sum())
    nes = NES(
        w=base_array,
        f=reward_func,
        pop=50,
        sigma=5 * 10 ** (-1),
        alpha=5 * 10 ** (-2)
    )
    loss_array = nes.optimize(n_iter=500, silent=False)
    print(nes.w)
    plt.plot(loss_array, label="loss")
    plt.show()
    