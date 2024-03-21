"""
Implementation of the NES (Natural Evolution Strategy)

# TODO: ==> DONE
To reduce variance, we use antithetic sampling Geweke [1988], 
also known as mirrored sampling Brockhoff et al. [2010] in the 
ES literature: that is, we always evaluate pairs of perturbations +eps, −eps,

# TODO:
softmax pour éviter les exploding gradients
"""
import numpy as np
from typing import Optional, Callable, Any, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt


class NES:

    def __init__(
            self,
            w: np.ndarray, # base matrix
            f: Callable[[np.ndarray], float], # reward function
            pop: int = 50, # population size
            sigma: float = 0.1, # noise standard deviation
            alpha: float = 0.001, # learning rate
            adaptive_rate: bool = False,
            mirrored_sampling: bool = False,
            f_test: Optional[Callable[[np.ndarray], float]] = None, # function to check testing loss
        ):
        """
        Args
            w: np.ndarray
                Array to optimize
            f: Callable[[np.ndarray] -> float]
                Reward function that will be maximized
            pop: int
                Population size
            sigma: float
                Spread noise of the new generation from the precendent
            alpha: float
                Learning rate
            adaptive_rate: bool
                Whether to adapt the learning rate
            mirrored_sampling: bool
                Replaces the estimation of f(w + e) by (f(w + e) - f(w - e)) / 2
                **THIS WILL DOUBLE THE COMPUTATION TIME FOR A GIVEN POPULATION**
            f_test: Optional[Callable[[np.ndarray], float]] = None
                Function that will be called at the end of each generation to measure
                the loss of the current model on the test set. Results saved in testing_loss

        """
        self.pop: int = pop
        self.sigma: float = sigma
        self.alpha: float = alpha
        self.w: np.ndarray = w.copy()
        self.f: Callable[[np.ndarray], float] = f
        self.f_test: Optional[Callable[[np.ndarray], float]] = f_test
        self.mirrored_sampling: bool = mirrored_sampling
        self.adaptive_rate: bool = adaptive_rate
        self.n_avg: int = 5

        self.data = pd.DataFrame(columns=["train_loss", "test_loss", "alpha", "noise"])
        
    def step(self):
        """
        Computes a step of the NES.
        
        Returns:
            float
                reward function applied to the matrix AFTER the step was carried
        """
        new_gen: List[np.ndarray] = [np.random.normal(0, 1, self.w.shape) for _ in range(self.pop)]

        rewards: np.ndarray = np.zeros(self.pop)
        for i, elem in enumerate(new_gen):
            # We compute the reward for our base point + sigma * delta[i]
            if self.mirrored_sampling:
                rewards[i] = 0.5 * (
                    self.f(self.w + self.sigma * elem) -
                    self.f(self.w - self.sigma * elem)
                )
            else:
                rewards[i] = self.f(self.w + self.sigma * elem)
        
        # Standardize the rewards to have a gaussian distribution
        rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        # The division by sigma has been removed below because it seemed that the normalization already
        # removed the influence of sigma
        rewards *= self.alpha / (self.pop) # * self.sigma)
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
            

    def optimize(
            self, 
            n_iter: int = 100, 
            silent: bool = False,
            graph: bool = False,
            ) -> np.ndarray:
        """
        Runs n_iter steps of the NES algorithm

        Args:
            n_iter: int
                Number of iterations
            silent: bool
                Whether to print results or not
            graph: bool
                Whether to show a graph of loss / learning rate / noise

        Returns:
            1D np.ndarray of the training loss at the end of each generation
        """
        window_size = 15

        pbar = tqdm(range(n_iter), disable=silent)
        for i in pbar:
            n = len(self.data)

            self.data.loc[n, "noise"] = self.sigma
            self.data.loc[n, "alpha"] = self.alpha
            self.data.loc[n, "train_loss"] = self.step()
            self.data.loc[:, "score"] = (self.data["train_loss"].diff() > 0).rolling(window_size).sum() / window_size
            # self.data.loc[:, "score"] = self.data["train_loss"].rolling(5).mean().diff(1).dropna() / self.data.loc[5:, "alpha"]
            if self.adaptive_rate:

                if self.data.loc[n, "score"] >= 0.95:
                    self.alpha *= 1.03
                elif self.data.loc[n, "score"] <= 0.75:
                    self.alpha /= 1.03
            if self.f_test is not None:
                self.data.loc[n, "test_loss"] = self.f_test(self.w)
                pbar.set_description(f"Current loss: {self.data.loc[n, 'train_loss']}, " +
                                     f"Test loss: {self.data.loc[n, 'test_loss'][i]}")
            else:
                pbar.set_description(f"Current loss: {self.data.loc[n, 'train_loss']}")

        if graph:  # If asked, we plot the graph
            self.plot(log=True)
        return self.data["train_loss"].iloc[-1]
    
    def plot(
            self, 
            log: bool = False,
            save_path: Optional[str] = None,
            optimal_value: Optional[float] = None
        ) -> None:

        fig, (ax1, ax2, ax4) = plt.subplots(3, 1, sharex=True)
        df: pd.DataFrame = self.data.copy()
        df["train_loss"] = np.abs(df["train_loss"].astype(float))
        df["alpha"] = df["alpha"].astype(float)
        df["noise"] = df["noise"].astype(float)
        ax1.plot(df.index, df.loc[:, "train_loss"], 'r-')
        if optimal_value is not None:
            ax1.plot(df.index, [optimal_value for _ in df.index], 'g-')
        # ax1.set_title('Loss')

        ax2.plot(df.index, df.loc[:, "alpha"], 'g-')
        # ax2.set_title('Learning rate')

        # ax3.plot(df.index, df.loc[:, "noise"], 'b-')
        # ax3.set_title('Noise')
        
        ax4.plot(df["score"].dropna().index, df["score"].dropna().abs(), "b-")
        ser = df["score"].dropna().abs()
        ser[ser > 0] = None
        ax4.plot(ser.index, ser, color="red")
        # ax4.set_yscale("log")
        
        if log:
            ax1.set_yscale("log")
            ax2.set_yscale("log")
            # ax3.set_yscale("log")
        
        ax1.grid(True)
        ax2.grid(True)
        # ax3.grid(True)
        ax4.grid(True)
        fig.text(0.5, 0.04, 'Step', ha='center', fontsize=12)

        # plt.title(f"Steps: {len(self.df)},")
        if save_path is not None:
            fig.savefig(save_path)

        plt.show()

    