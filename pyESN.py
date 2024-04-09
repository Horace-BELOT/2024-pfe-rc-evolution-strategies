"""
The dimensions of the matrix are:
- W_in (input layer matrix) => n_reservoir x n_inputs
- W (reservoir connectivity matrix) => n_reservoir x n_reservoir
- W_out (output layer) => n_outputs x n_reservoir

The update formula is:
x[n + 1] = activation(W * x[n] + W_in * u(n + 1) + W_fb * y[n])

"""

import numpy as np
import tqdm
import pickle
from sklearn.base import BaseEstimator

from utils import * 
from RLS import RLS
from typing import Callable, Literal, Optional, Union, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy


Learning_Method = Literal["pinv", "pinv_ridge", "sgd", "sgd_ridge", "custom"]

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
                 custom_method: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
                 ridge_noise: Optional[float] = None, learning_rate: float = 0,
                 input_to_output_ratio: int = 1, allow_cut_connections: bool = False,
                 repeated_inputs: Optional[int] = None,
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
            input_to_output_ratio: for MNIST, handles the case where you feed col by col
                instead of the entire image. The integer counts how many entries match 1 output.
                By default, there is 1 input per output
            allow_cut_connections: whether or not to allow the reservoir to have connections from input to output
            repeated_inputs: how many times every input is passed in succession
        
        [Scaling]
            input_scaling: factor that input weights array W_in will be multiplied by
            feedback_scaling: factor that output weights array W_fb will be multiplied by


        [Training]
            learn_method: "pinv", "pinv_ridge", "sgd", "sgd_ridge", "custom"
            custom_method: if method is custom, this shouuld be a method such that custom_method(x,y) = w
            learning_rate: learning rate used for SGD
            ridge_noise: ridge regression noise (regularization)

        [Misc]
            silent: whether or not to print updates on execution
            random_state: seed for reservoir, input layer and feedback layer initialization
            
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
        self.input_to_output_ratio: int = input_to_output_ratio
        self.allow_cut_connections: bool = allow_cut_connections
        self.repeated_inputs: Optional[int] = repeated_inputs

        # Model
        self.learn_method: Learning_Method = learn_method
        self.custom_method: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = custom_method
        if self.learn_method == "custom" and self.custom_method is None:
            raise ValueError("Can't have learn_method = custom without providing a custom_method")
        self.ridge_noise: Optional[float] = ridge_noise
        self.learning_rate: float = learning_rate

        # Misc
        self.silent: bool = silent  # Whether or not to print things / show tqdm bar
        self.random_state_: np.random.RandomState
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif isinstance(random_state, int):
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:  # random_state is None
            self.random_state_ = np.random.mtrand._rand

        # Network components
        self.W: np.ndarray  # Reservoir connectivity matrix
        self.W_in: np.ndarray  # Input layer matrix
        self.W_out: np.ndarray  # Output layer matrix (shape = n_outputs x n_reservoir)
        self.W_fb: np.ndarray  # Feedback connectivity array
        self.states: np.ndarray  = None # list of consecutive reservoir states (N x n_reservoir)
        self.extended_states: np.ndarray = None  # states + inputs (N x (n_reservoir + n_inputs))

        self.build_matrixes()

    def to_dict(self) -> Dict[str, Any]:
        """
        Builds a dictionnary containing all the elements of the ESN object
        """
        return {
            "args": {
                "n_inputs": self.n_inputs,
                "n_outputs": self.n_outputs,
                "n_reservoir": self.n_reservoir,
                "spectral_radius": self.spectral_radius,
                "sparsity": self.sparsity,
                "leaky_rate": self.leaky_rate,
                "noise": self.noise,
                "state_activ_fx": self.state_activ_fx,
                "out_activ": self.out_activ,
                "out_activ_inv": self.out_activ_inv,
                "wash_out": self.wash_out,
                "input_scaling": self.input_scaling,
                "feedback_scaling": self.feedback_scaling,
                "input_to_output_ratio": self.input_to_output_ratio,

                "learn_method": self.learn_method,
                "custom_method": self.custom_method,
                "ridge_noise": self.ridge_noise,
                "learning_rate": self.learning_rate,

                "silent": self.silent,
            },
            "objects": {
                "random_state_": self.random_state_,
                "W": self.W,
                "W_in": self.W_in,
                "W_out": self.W_out,
                "W_fb": self.W_fb,
                "states": self.states,
                "extended_states": self.extended_states,
            }
        }

    def save(self, file_path: str) -> None:
        """
        Saves the content of the ESN object to a pickle fle
        """
        if ".pickle" not in file_path:
            file_path = file_path + ".pickle"
        with open(file_path, "wb") as handle:
            pickle.dump(self.to_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def from_dict(data: Dict[Any, Any]) -> "ESN":
        """"""
        esn = ESN(**data["args"])
        if data["objects"].get("random_state_") is not None:
            esn.random_state_ = data["objects"].get("random_state_")
        if data["objects"].get("W") is not None:
            esn.W = data["objects"].get("W")
        if data["objects"].get("W_in") is not None:
            esn.W_in = data["objects"].get("W_in")
        if data["objects"].get("W_out") is not None:
            esn.W_out = data["objects"].get("W_out")
        if data["objects"].get("W_fb") is not None:
            esn.W_fb = data["objects"].get("W_fb")
        if data["objects"].get("states") is not None:
            esn.states = data["objects"].get("states")
        if data["objects"].get("extended_states") is not None:
            esn.extended_states = data["objects"].get("extended_states")
        return esn

    def load(file_path: str) -> "ESN":
        """
        Loads an ESN object from 
        """
        with open(file_path, "rb") as handle:
            loaded_data: Dict[str, Any] = pickle.load(handle)

        return ESN.from_dict(loaded_data)
        
    def copy():
        """
        Create an identical ESN object with deep copies
        """
        raise NotImplementedError


    def build_matrixes(self):
        """
        Builds the matrix components:
         - Input layer matrix
         - Reservoir connectivity matrix
         - Feedback matrix
        """
        ## Builds the reservoir matrix
        # Initiate the connectivity matrix with a uniform law in [-0.5, 0.5]
        self.W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # We force the sparsity of the matrix to the given value
        self.W[self.random_state_.rand(self.n_reservoir, self.n_reservoir) < self.sparsity] = 0
        # We then force the spectral radius to the given value
        radius_w: float = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W = self.W * (self.spectral_radius / radius_w)

        ## Builds the input matrix
        # Dimensions are (n_reservoir, n_inputs)
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.W_in *= self.input_scaling

        ## Builds the feedback matrix
        # Dimensions are (n_reservoir, n_outputs)
        self.W_fb = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1
        self.W_fb *= self.feedback_scaling

        ## Initialize the output layer
        if self.allow_cut_connections:
            self.W_out = np.zeros(shape=(self.n_outputs, (self.n_inputs + self.n_reservoir)))
        else:
            self.W_out = np.zeros(shape=(self.n_outputs, self.n_reservoir))

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
        
    def feed(
            self, 
            inputs: np.ndarray, 
            outputs: Optional[np.ndarray] = None,
            build_outputs: bool = False,
            wash_out: Optional[int] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feeds inputs and outputs (through feedback loop) into the reservoir.

        Args:
            inputs: np.ndarray of inputs of shape (N x n_inputs)
            outputs: Optional[np.ndarray] of outputs of shape (N x n_outputs).
                If None: initialized as zeros
            build_outputs: bool. Whether or not the ESN will build the outputs using
                the states, the activation function and the current W_out matrix.
                - If True: then the feedback component of step n + 1 will be built 
                    from current W_out and state n.
                - If False: then the feedback component will be built from provided
                    outputs. Otherwise matrices of zeros will be used (no feedback).
            wash_out: Optional[int] can override the wash_out parameter of ESN

        Returns:
            np.ndarray of extended states (states + inputs concatenated) of 
                shape (N x (n_inputs + n_reservoir))
            np.ndarray of outputs
        """
        n_truncate: int = wash_out if wash_out is not None else self.wash_out
        ### First we need to reshape potentially ill-shaped inputs.
        if inputs.ndim < 2: inputs = np.reshape(inputs, (len(inputs), -1))
        if inputs.shape[1] != self.n_inputs: 
            raise ValueError(f"Inputs are of wrong shape: {inputs.shape} instead of (N, {self.n_inputs})")
        n: int = inputs.shape[0]
        
        if outputs is None:  # If no outputs is given
            outputs = np.zeros(shape=(n, self.n_outputs))
        else:
            if outputs.ndim < 2: outputs = np.reshape(outputs, (len(outputs), -1))
            if outputs.shape[1] != self.n_outputs: 
                raise ValueError(f"Inputs are of wrong shape: {inputs.shape} instead of (N, {self.n_outputs})")
            # Checking if inputs and outputs have the same size
            if outputs.shape[0] != inputs.shape[0]:
                raise ValueError(f"Inputs and Outputs have different shapes ({outputs.shape[0]} != {inputs.shape[0]})")
        
        ### Feeding the network and harvesting states
        if not self.silent: 
            print("[INFO] Harvesting states...")

        
        states: np.ndarray = np.zeros(shape=(n, self.n_reservoir))
        outputs = outputs.copy()  # This is to avoid editing the matrix given in function input

        progress_bar = tqdm.tqdm(range(n - 1), disable=self.silent)  # Progress bar
        for k in progress_bar:
            states[k + 1, :] = self._update(states[k], inputs[k + 1, :], outputs[k, :])
            if self.repeated_inputs is not None:
                # If asked for, each inputs will be fed multiple times to the network before
                # the output is harvested
                for _ in range(1, self.repeated_inputs):
                    states[k + 1, :] = self._update(states[k + 1], inputs[k + 1, :], outputs[k, :])

            if build_outputs:   
                # If no output array was given, then we build the output array on the fly
                # This is necessary as the feedback loop needs the output for time-dependant
                # problems
                
                if self.allow_cut_connections:
                    outputs[k + 1, :] = self.out_activ(np.dot(
                    self.W_out, np.concatenate([states[k + 1, :], inputs[k + 1, :]])))
                else:
                    outputs[k + 1, :] = self.out_activ(np.dot(self.W_out, states[k + 1, :]))
                

            # Keeping track of max coeff to be sure that no divergence occurs
            # max_res: float = np.abs(self.states[k, :]).max()
            # progress_bar.set_description(f"Step {k}/{n}. Max in reservoir = {max_res:.2f}")
                
        self.states = states
        x: np.ndarray = None
        if self.allow_cut_connections:
            x: np.ndarray = np.hstack([self.states, inputs])  # inputs of training is states + inputs
        else:
            x: np.ndarray = self.states
        self.extended_states = x
        if not build_outputs:
            # If outputs were not built on the go => we build them at the end
            for k in range(n - 1):
                outputs[k + 1, :] = self.out_activ(np.dot(self.W_out, x[k + 1]))

            
        y: np.ndarray = self.out_activ_inv(outputs[n_truncate:, :])
        if self.input_to_output_ratio != 1:
            t: int = self.input_to_output_ratio
            return x[n_truncate + (t - 1)::t, :], outputs[n_truncate + (t - 1)::t, :]
        return x[n_truncate:, :], outputs[n_truncate:, :]


    def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        """
        Feeds all the inputs to the network and harvests the resulting states.
        Then trains the model using the chosen method and fitting the output on the target.
        Finally, returns the prediction post-training of the ESN model on the train set.

        Args:
            inputs: np.ndarray of inputs of shape (N x n_inputs)
            outputs: np.ndarray of outputs of shape (N x n_outputs)
        Returns:
            np.ndarray representing the prediction of the model on the trainset (N x n_outputs)
        """
        if inputs.ndim < 2: inputs = np.reshape(inputs, (len(inputs), -1))
        if inputs.shape[1] != self.n_inputs: 
            raise ValueError(f"Inputs are of wrong shape: {inputs.shape} instead of (N, {self.n_inputs})")
        n: int = inputs.shape[0]
        
        if outputs is None:  # If no outputs is given
            outputs = np.zeros(shape=(n, self.n_outputs))
        else:
            if outputs.ndim < 2: outputs = np.reshape(outputs, (len(outputs), -1))
            if outputs.shape[1] != self.n_outputs: 
                raise ValueError(f"Inputs are of wrong shape: {inputs.shape} instead of (N, {self.n_outputs})")
            # Checking if inputs and outputs have the same size
            if outputs.shape[0] != inputs.shape[0]:
                raise ValueError(f"Inputs and Outputs have different shapes ({outputs.shape[0]} != {inputs.shape[0]})")
        
        x, _ = self.feed(inputs, outputs, wash_out=self.wash_out)
        t: int = self.input_to_output_ratio
        y: np.ndarray = outputs[self.wash_out + (t - 1)::t, :]

        ### Training output matrix (readout layer)
        if not self.silent: print("[INFO] Training Readout Layer...")
        ## Training data
        if self.learn_method == "pinv":
            self.W_out = pinv(x, y)
        elif self.learn_method == "pinv_ridge":
            self.W_out = pinv_ridge(x, y, self.ridge_noise)
        elif self.learn_method == "sgd":
            self.W_out = sgd(x, y, alpha=self.learning_rate, lambda_ridge=0, silent=self.silent)
        elif self.learn_method == "sgd_ridge":
            self.W_out = sgd(x, y, alpha=self.learn_method, lambda_ridge=self.ridge_noise, silent=self.silent)
        elif self.learn_method == "custom":
            self.W_out = self.custom_method(x,y)
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
            # TODO: implement
            last_input: np.ndarray = np.zeros(self.n_inputs)
            last_state: np.ndarray = np.zeros(self.n_reservoir)
            last_output: np.ndarray = np.zeros(self.n_outputs)
        else:
            last_input: np.ndarray = np.zeros(self.n_inputs)
            last_state: np.ndarray = np.zeros(self.n_reservoir)
            last_output: np.ndarray = np.zeros(self.n_outputs)
        inputs = np.vstack([last_input, inputs])
        states = np.vstack([last_state, np.zeros([n, self.n_reservoir])])
        outputs = np.vstack([last_output, np.zeros([n, self.n_outputs])])

        x, y = self.feed(inputs=inputs, outputs=None, wash_out=1)
        # for k in tqdm(range(n), disable=self.silent):
        #     states[k + 1, :] = self._update(states[k, :], inputs[k + 1, :], outputs[k, :])
        #     # outputs[k + 1, :] = self.out_activ(np.dot(
        #     #     self.W_out, np.concatenate([states[k + 1, :], inputs[k + 1, :]])))
        # self.states = states
        # self.extended_states = np.hstack([states[1:, :], inputs[1:, :]])
        # return self.out_activ(np.dot(self.extended_states, self.W_out.T))
        t: int = self.input_to_output_ratio
        return y[(t-1)::t]
        # return self.out_activ(outputs[1:])

# Now the implementation in pytorch

class ESN_State_Layer(nn.Module):
    
    def __init__(self,n_reservoir: int = 50,
                 spectral_radius: float = 0.95, sparsity: float = 0,
                 leaky_rate: float = 0, noise: float = 0.0, 
                 state_activ_fx: Callable[[np.ndarray], np.ndarray] = torch.tanh,
                 device: str = "cpu"
                 ):
        """
        Args:
        [Network's parameter]
            n_reservoir: number of reservoir neurons
            spectral_radius: spectral radius of connectivity matrix of the reservoir
            sparsity: proportion of recurrent weights set to zero in the reservoir
            noise: noise added to each neuron when updating (regularization)
            state_activ_fx: activation function used in updating reservoir states
            leaky_rate: leaky rate of Leaky-Integrator ESN (LIESN), used to improve STM
            device: device on which the model will be run
        """
        super().__init__()
        self.n_reservoir: int = n_reservoir
        self.spectral_radius: float = spectral_radius
        self.sparsity: float = sparsity
        self.leaky_rate: float = leaky_rate
        self.noise: float = noise
        self.state_activ_fx: Callable[[np.ndarray], np.ndarray] = state_activ_fx
        self.device: str = device
        self.W: nn.Parameter
        self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initializes the weights of the reservoir
        """
        ## Builds the reservoir matrix
        # Initiate the connectivity matrix with a uniform law in [-0.5, 0.5]
        self.W = torch.rand(self.n_reservoir, self.n_reservoir, requires_grad=False, device=self.device) - 0.5
        # We force the sparsity of the matrix to the given value
        self.W[torch.rand(self.n_reservoir, self.n_reservoir, device=self.device) < self.sparsity] = 0
        # We then force the spectral radius to the given value
        radius_w: float = torch.max(torch.abs(torch.linalg.eigvals(self.W)))
        self.W = self.W * (self.spectral_radius / radius_w)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: torch.Tensor of current reservoir state (length of array is n_reservoir)
        Returns:
            the reservoir updated state after being fed the given inputs
        """
        preactivation: torch.Tensor = torch.matmul(self.W, state)
        # We are going to use a normal white noise matrix
        noise_matrix: torch.Tensor = torch.normal(mean=0, std=self.noise, size=(self.n_reservoir,), requires_grad=False, device=self.device)
        return self.leaky_rate * state + noise_matrix + self.state_activ_fx(preactivation)

class Torch_ESN(nn.Module):
        
    def __init__(self, n_inputs: int, n_outputs: int, n_reservoir: int = 50,
                spectral_radius: float = 0.95, sparsity: float = 0,
                leaky_rate: float = 0, noise: float = 0.0, 
                state_activ_fx: Callable[[np.ndarray], np.ndarray] = torch.tanh, 
                out_activ: Callable[[np.ndarray], np.ndarray] = torch.nn.Identity(),
                out_activ_inv: Callable[[np.ndarray], np.ndarray] = identity,
                input_scaling: float = 1, feedback_scaling: float = 0.0,
                seed: int = None, wash_out: int = 1, silent: bool = True,
                learning_rate: float = 0.001, nb_epochs: int = 5,
                batch_size: int = 32,
                input_to_output_ratio: int = 1, allow_cut_connections: bool = False,
                device: str = "cpu"
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
            input_to_output_ratio: for MNIST, handles the case where you feed col by col
                instead of the entire image. The integer counts how many entries match 1 output.
                By default, there is 1 input per output
            allow_cut_connections: whether or not to allow the reservoir to have connections from input to output
        
        [Scaling]
            input_scaling: factor that input weights array W_in will be multiplied by
            feedback_scaling: factor that output weights array W_fb will be multiplied by


        [Training]
            learning_rate: learning rate used for ADAM
            nb_epochs: number of epochs for training
            batch_size: size of the batch for training
            
        [Misc]
            silent: whether or not to print updates on execution
            seed: seed for reservoir, input layer and feedback layer initialization
            device: device on which the model will be run
            
        """
        super().__init__()
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
        self.input_to_output_ratio: int = input_to_output_ratio
        self.allow_cut_connections: bool = allow_cut_connections
        
        # Model
        self.learning_rate: float = learning_rate
        self.nb_epochs: int = nb_epochs
        self.batch_size: int = batch_size
        
        # Misc
        self.silent: bool = silent
        self.seed = seed
        torch.manual_seed(seed)
        self.device = device
        
        
        self.layer_out: nn.Linear = nn.Linear(n_reservoir, n_outputs, bias=False)
        
        with torch.no_grad():
            self.state_layer = ESN_State_Layer(n_reservoir=n_reservoir, spectral_radius=spectral_radius,
                                            sparsity=sparsity, leaky_rate=leaky_rate, noise=noise,
                                            state_activ_fx=state_activ_fx, device=self.device)
            print(n_inputs, n_reservoir, n_outputs)
            self.layer_in: nn.Linear = nn.Linear(n_inputs, n_reservoir, bias=False)
            self.layer_in.requires_grad_(False)
            print(self.layer_in.weight.shape)
            self.feedback_layer: nn.Linear = nn.Linear(n_outputs, n_reservoir, bias=False)
            self.feedback_layer.requires_grad_(False)
            self.cut_connections: nn.Linear = nn.Linear(n_inputs, n_outputs, bias=False)
            self.cut_connections.requires_grad_(False)
        
        #self.initialize_weights()
    
    def initialize_weights(self):
        """
        Initializes the weights of the reservoir
        """
        self.layer_in.weight.data.uniform_(-1, 1)
        self.layer_out.weight.data.uniform_(-1, 1)
        self.feedback_layer.weight.data.zero_()
        self.cut_connections.weight.data.zero_()
    
    def forward(self, inputs: torch.Tensor, outputs: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feeds inputs and outputs (through feedback loop) into the reservoir.
        
        Args:
            inputs: torch.Tensor of inputs of shape (N x n_inputs)
            outputs: Optional[torch.Tensor] of outputs of shape (N x n_outputs).
                If None: initialized as zeros
        Returns:
            output array of the network
        """
        n: int = inputs.shape[0]
        
        if outputs is None:
            outputs = torch.zeros(n, self.n_outputs, device=self.device)
        else:
            if outputs.shape[0] != n:
                raise ValueError(f"Inputs and Outputs have different shapes ({outputs.shape[0]} != {inputs.shape[0]})")
        
        if state is None:
            state = torch.zeros(self.n_reservoir, device=self.device)
        else:
            if state.shape[0] != self.n_reservoir:
                raise ValueError(f"State is of wrong shape: {state.shape} instead of ({self.n_reservoir},)")
            state = state.clone()
            
        
        
        outputs = outputs.clone()
        
        for k in range(n - 1):
            inp =  self.layer_in(inputs[k + 1, :])
            feed = self.feedback_layer(outputs[k, :])
            st = self.state_layer(state)
            next_state = st + inp + feed
            if self.allow_cut_connections:
                next_output = self.out_activ(self.layer_out(next_state)) + self.cut_connections(inputs[k + 1, :])
            else:
                next_output = self.out_activ(self.layer_out(next_state))
            state = next_state.detach().clone()
            outputs = torch.cat((outputs[:k+1], next_output.unsqueeze(0)))
        
        t: int = self.input_to_output_ratio
        return outputs[(t - 1)::t, :], state
    
    # fit the layer_out with SGD
    def fit(self, inputs: torch.Tensor, outputs: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Feeds all the inputs to the network and harvests the resulting states.
        Then trains the model using the chosen method and fitting the output on the target.
        Finally, returns the prediction post-training of the ESN model on the train set.

        Args:
            inputs: torch.Tensor of inputs of shape (N x n_inputs)
            outputs: torch.Tensor of outputs of shape (N x n_outputs)
        Returns:
            torch.Tensor representing the prediction of the model on the trainset (N x n_outputs)
        """
        
        ### Training output matrix (readout layer)
        if not self.silent: print("[INFO] Training Readout Layer...")
        optimizer = torch.optim.Adam(self.layer_out.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # detect gradient anomaly
        torch.autograd.set_detect_anomaly(True)
        if state is None:
            state = torch.zeros(self.n_reservoir, device=self.device)
        else:
            if state.shape[0] != self.n_reservoir:
                raise ValueError(f"State is of wrong shape: {state.shape} instead of ({self.n_reservoir},)")
            state = state.clone()
        
        for epoch in range(self.nb_epochs):
            _ = self.forward(inputs[:self.wash_out, :], outputs[:self.wash_out, :])
            nb_batch = (inputs.shape[0]-self.wash_out) // self.batch_size
            accuracies = []
            for i in range(nb_batch):
                optimizer.zero_grad()
                inputs_batch = inputs[self.wash_out + i*self.batch_size:self.wash_out + (i+1)*self.batch_size, :]
                outputs_batch = outputs[self.wash_out + i*self.batch_size:self.wash_out + (i+1)*self.batch_size, :]
                pred,state = self.forward(inputs_batch, outputs_batch, state)
                loss = criterion(pred, outputs_batch)
                accuracy = (pred.argmax(dim=1) == outputs_batch.argmax(dim=1)).float().mean()
                accuracies.append(accuracy.item())
                loss.backward(retain_graph=True)
                optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch {epoch} - Batch {i} - Loss: {loss.item()} - Accuracy: {np.mean(accuracies)}")
            
            print(f"Epoch {epoch} - Loss: {loss.item()} - Accuracy: {np.mean(accuracies)}")
        
        return pred, state
    
    def set_W_in(self, W_in: torch.Tensor):
        self.layer_in.weight.data = W_in