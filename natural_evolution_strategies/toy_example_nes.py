"""
The goal of this code is to make an oversimplified example of a NES to have a view on 
how the system functions and how to implement it.
"""
from typing import Callable
import numpy as np
import sys
import os

from sklearn.linear_model import SGDClassifier
try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass

class ToyExampleNES:
    def __init__(
            self,
            f: Callable[[np.ndarray], np.ndarray], # target function to optimize
            dim: int, # dimension of the problem
            npop: int = 50, # population size
            sigma: float = 0.1, # noise standard deviation
            alpha: float = 0.001, # learning rate
        ):
        self.npop = npop
        self.sigma = sigma
        self.alpha = alpha
        self.dim = dim
        self.f = f

        self.w = np.zeros(self.dim)

    def _step(self):
        """Does a step of the algorithm"""
        # Samples npop new vectors from normal distribution N(0,1)
        # if dim = 3, then we sample eg 50 vectors in R^3
        # from there, we will compute the gradient between the original point and
        # these new 50 points.
        N = np.random.randn(self.npop, self.dim)
        # For each member of this new generation we compute the reward
        # We then build the vector R where R[j] is the evaluation of the reward function
        # at the point:         w + sigma * N[j]
        R = np.array([self.f(self.w + self.sigma * n_j) for n_j in N])

        # We standardize the rewards to have a gaussian distribution
        A = (R - np.mean(R)) / np.std(R)

        # We then update the current guess matrix 
        self.w = self.w + self.alpha / (self.npop * self.sigma) * np.dot(N.T, A)
        

    def run(self, n_iter: int = 100, verbose=True):
        """
        Runs the optimization for n_iter
        """
        
        for i in range(n_iter):
            self._step()
            print(f"Current loss at step {i+1}/{n_iter}: {self.f(self.w)}")





if __name__ == "__main__":
    # We will attempt to optimize the L2 loss to find a target vector in R3
    dim: int = 5
    target = np.random.randn(dim)  # The target vector is chosen randomly.
    f_loss = lambda x, target=target: -np.sum(np.square(x - target)) # func to maximize

    nes = ToyExampleNES(
        f=f_loss,
        dim=dim,
        npop=50, 
        sigma=0.1, 
        alpha=0.001)
    
    nes.run(n_iter=500, verbose=True)
