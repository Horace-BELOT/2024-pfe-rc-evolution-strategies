"""
The dimensions of the matrix are:
- W_in (input layer matrix) => n_reservoir x n_inputs
- W (reservoir connectivity matrix) => n_reservoir x n_reservoir
- W_out (output layer) => n_outputs x n_reservoir

The update formula is:
x[n + 1] = activation(W * x[n] + W_in * u(n + 1) + W_fb * y[n])

"""

import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator
from utils import * 
from RLS import RLS
from typing import Callable, Literal, Optional

Learning_Method = Literal["pinv", "pinv_ridge", "sgd", "sgd_ridge"]

class ESN:

    def __init__(self, n_inputs: int, n_outputs: int, n_reservoir: int = 50,
                 spectral_radius: float = 0.95, sparsity: float = 0,
                 leaky_rate: float = 0, noise: float = 0.0, 
                 state_activ_fx: Callable[[np.ndarray], np.ndarray] = np.tanh, 
                 out_activ: Callable[[np.ndarray], np.ndarray] = identity,
                 out_activ_inv: Callable[[np.ndarray], np.ndarray] = identity,
                 input_scaling: float = 1, feedback_scaling: float = 0.0,
                 random_state: int = None, wash_out: int = 1, silent: bool = True,
                 learn_method: Learning_Method = "pinv",
                 ridge_noise: Optional[float] = None, learning_rate: float = 0,
                 ):
        """
        Args:
        [Network's parameter]
            n_inputs: input dimensions
            n_outputs: output dimensions
            n_reservoir: number of reservoir neurons
            spectral_radius: spectral radius of connectivity matrix of the reservoir
            sparsity: proportion of recurrent weights set to zero in the reservoir
            noise: noise added to each neuron when updating (regularization)
            wash_out: number of states that we will discard at the start of the output
            state_activ_fx: activation function used in updating reservoir states
            out_activ: output activation function (often identity or softmax)
            out_activ_inv: inverse of the output activation function
            leaky_rate: leaky rate of Leaky-Integrator ESN (LIESN), used to improve STM
        
        [Scaling]

        [Training]
            learn_method: "pinv", "pinv_ridge", "sgd", "sgd_ridge"
            ridge_noise: ridge regression noise (regularization)
            
        """
        # Constants / network characteristics
        self.n_inputs: int = n_inputs
        self.n_outputs: int = n_outputs
        self.n_reservoir: int = n_reservoir
        self.spectral_radius: float = spectral_radius
        self.sparsity: float = sparsity
        self.leaky_rate: float = leaky_rate
        self.noise: float = noise
        self.state_activ_fx: Callable[[np.ndarray], np.ndarray] = state_activ_fx
        self.out_activ: Callable[[np.ndarray], np.ndarray] = out_activ
        self.out_activ_inv: Callable[[np.ndarray], np.ndarray] = out_activ_inv
        self.wash_out: int = wash_out
        self.input_scaling: float = input_scaling
        self.feedback_scaling: float = feedback_scaling

        # Model
        self.learn_method: Literal["pinv", "ridge"] = learn_method
        self.ridge_noise: Optional[float] = ridge_noise
        self.learning_rate: float = learning_rate

        # Network components
        self.W: np.ndarray  # Reservoir connectivity matrix
        self.W_in: np.ndarray  # Input layer matrix
        self.W_out: np.ndarray  # Output layer matrix (shape = n_outputs x )
        self.W_fb: np.ndarray  # Feedback connectivity array
        self.states: np.ndarray  # list of consecutive reservoir states (N x n_reservoir)
        self.extended_states: np.ndarray  # states + inputs (N x (n_reservoir + n_inputs))

        # Misc
        self.silent: bool = silent  # Whether or not to print things / show tqdm bar

        self.build_matrixes()

    def build_matrixes(self):
        """
        Builds the matrix components:
         - Input layer matrix
         - Reservoir connectivity matrix
         - Feedback matrix
        """
        ## Builds the reservoir matrix
        # Initiate the connectivity matrix with a uniform law in [-0.5, 0.5]
        self.W: np.ndarray = np.random.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # We force the sparsity of the matrix to the given value
        self.W[np.random.rand(self.n_reservoir, self.n_reservoir) < self.sparsity] = 0
        # We then force the spectral radius to the given value
        radius_w: float = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W = self.W * (self.spectral_radius / radius_w)

        ## Builds the input matrix
        # Dimensions are (n_reservoir, n_inputs)
        self.W_in = np.random.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.W_in *= self.input_scaling

        ## Builds the feedback matrix
        # Dimensions are (n_reservoir, n_outputs)
        self.W_fb = np.random.rand(self.n_reservoir, self.n_outputs) * 2 - 1
        self.W_fb *= self.feedback_scaling

        ## Initialize the output layer
        self.W_out = np.zeros(shape=(self.n_outputs, self.n_inputs))

    def _update(self, state: np.ndarray, input: np.ndarray, output: np.ndarray) -> np.ndarray:
        """
        Args:
            state: np.ndarray of current reservoir state (length of array is n_reservoir)
            input: np.ndarray of inputs (shape = (n_inputs, 1))
        
        Returns:
            the reservoir updated state after being fed the given inputs
        """
        preactivation: np.ndarray = (
            np.dot(self.W, state) + 
            np.dot(self.W_in, input) + 
            np.dot(self.W_fb, output)
        )
        # We are going to use a normal white noise matrix
        noise_matrix: np.ndarray = np.random.normal(scale=self.noise, size=self.n_reservoir)
        return self.leaky_rate * state + noise_matrix + self.state_activ_fx(preactivation)
        
    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        """
        Feeds all the inputs to the network and harvests the resulting states

        Args:
            inputs: np.ndarray of inputs of shape (N x n_inputs)
            outputs: np.ndarray of outputs of shape (N x n_outputs)
        Returns:

        """
        ### First we need to reshape potentially ill-shaped inputs.
        if inputs.ndim < 2: inputs = np.reshape(inputs, (len(inputs), -1))
        if inputs.shape[1] != self.n_inputs: 
            raise ValueError(f"Inputs are of wrong shape: {inputs.shape} instead of (N, {self.n_inputs})")
        if outputs.ndim < 2: outputs = np.reshape(outputs, (len(outputs), -1))
        if outputs.shape[1] != self.n_outputs: 
            raise ValueError(f"Inputs are of wrong shape: {inputs.shape} instead of (N, {self.n_outputs})")
        # Checking if inputs and outputs have the same size
        if outputs.shape[0] != inputs.shape[0]:
            raise ValueError("Inputs and Outputs have different shapes ()")
        n: int = inputs.shape[0]

        ### Feeding the network and harvesting states
        if not self.silent: print("[INFO] Harvesting states...")
        self.states: np.ndarray = np.zeros(shape=(n, self.n_reservoir))
        pbar  = tqdm(range(1, n), disable=self.silent)  # Progress bar
        for k in pbar:
            self.states[k, :] = self._update(self.states[k - 1], inputs[k, :], outputs[k - 1, :])
            # Keeping track of max coeff to be sure that no divergence occurs
            # max_res: float = np.abs(self.states[k, :]).max()
            # pbar.set_description(f"Step {k}/{n}. Max in reservoir = {max_res:.2f}")

        ### Training output matrix (readout layer)
        if not self.silent: print("[INFO] Training Readout Layer...")
        ## Preparing data
        x: np.ndarray = np.hstack([self.states, inputs])  # inputs of training is states + inputs
        self.extended_states = x
        x = x[self.wash_out:, :]  # removing first steps
        y: np.ndarray = self.out_activ_inv(outputs[self.wash_out:, :])

        ## Training data
        if self.learn_method == "pinv":
            self.W_out = pinv(x, y)
        elif self.learn_method == "pinv_ridge":
            self.W_out = ridge(x, y, self.ridge_noise)
        elif self.learn_method == "sgd":
            self.W_out = sgd(x, y, alpha=self.learning_rate, lambda_ridge=0, silent=self.silent)
        elif self.learn_method == "sgd_ridge":
            self.W_out = sgd(x, y, alpha=self.learn_method, lambda_ridge=self.ridge_noise, silent=self.silent)
        
        ## Predicting (we need to the full states without washout)
        pred_train: np.ndarray = self.out_activ(np.dot(self.extended_states, self.W_out.T))
        return pred_train
    
    def predict(self, inputs: np.ndarray, continuation: bool = False) -> np.ndarray:
        """
        Use the current network to predict the output for the given inputs.

        Args:
            inputs: np.ndarray of dimensions (N_test_samples x n_inputs)
            continuation: if True, start the network from the last training state
        Returns
            output array of the network
        """
        if inputs.ndim < 2: inputs = np.reshape(inputs, (len(inputs), -1))
        if inputs.shape[1] != self.n_inputs: 
            raise ValueError(f"Inputs are of wrong shape: {inputs.shape} instead of (N, {self.n_inputs})")
        n: int = np.shape(inputs)[0]

        if continuation:
            pass
        else:
            last_input: np.ndarray = np.zeros(self.n_inputs)
            last_state: np.ndarray = np.zeros(self.n_reservoir)
            last_output: np.ndarray = np.zeros(self.n_outputs)
        inputs = np.vstack([last_input, inputs])
        states = np.vstack([last_state, np.zeros([n, self.n_reservoir])])
        outputs = np.vstack([last_output, np.zeros([n, self.n_outputs])])

        for k in tqdm(range(n), disable=self.silent):
            states[k + 1, :] = self._update(self.states[k], inputs[k + 1, :], outputs[k, :])
            # outputs[k + 1, :] = self.out_activ(np.dot(
            #     self.W_out, np.concatenate([states[k + 1, :], inputs[k + 1, :]])))
        extended_states = np.hstack([states[1:, :], inputs[1:, :]])
        return self.out_activ(np.dot(extended_states, self.W_out.T))
        # return self.out_activ(outputs[1:])


    

