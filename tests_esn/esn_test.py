"""
This file includes test in a notebook-style fashion in order to explore the ESN capabilities
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
from utils import MnistDataloader
import matplotlib.pyplot as plt
import numpy as np


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    n, _ = y_pred.shape
    m, _ = y_true.shape
    if n != m:
        raise ValueError(f"Predicted data and True data dont have the same dimension: {n} != {m}")
    labels_pred: np.ndarray = np.array([np.argmax(y_pred[k]) for k in range(n)])
    labels_true: np.ndarray = np.array([np.argmax(y_true[k]) for k in range(n)])
    acc: float = np.sum(labels_pred == labels_true) / n
    return acc


def generate_mackey_glass(args: dict):
    from jitcdde import jitcdde, y, t
    import numpy as np
    τ = args["mg_tau"]
    n = args["mg_n"]
    β = args["mg_beta"]
    γ = args["mg_gamma"]
    history = args["mg_history"]
    stepsize = args["mg_T"]  # fixed for the moment (check, if smaller values are required)
    length = args["series_length"]

    f = [β * y(0, t - τ) / (1 + y(0, t - τ) ** n) - γ * y(0)]
    DDE = jitcdde(f)
    DDE.set_integration_parameters(atol=1.0e-16, rtol=1.0e-10)  # min_step = 1.0e-15

    DDE.constant_past([history])
    # DDE.step_on_discontinuities()
    DDE.integrate_blindly(0.0)  # This gives the results comparable to the MATLAB dde23 solver

    data = []
    for time in np.arange(DDE.t, DDE.t + length, stepsize):
        data.append(DDE.integrate(time))

    return np.array(data).squeeze()

class DataLoader:
    
    def load_mnist():
        """Loads the MNIST dataset"""
        input_path = 'data'
        training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data()
        return (x_train, y_train), (x_test, y_test)
    
    def periodic_signal(length: int = 1000, period: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        
        """
        x: np.ndarray = np.ones(length)
        y: np.ndarray = np.zeros(length)
        y[[k for k in range(length) if k % period == 0]] = 1
        return x, y

    def delay_timeseries(n: int = 5, length: int = 10000, pct_signal: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates a timeseries where if x[i] = 1, y[i + n] = 1 otherwise x and y = 0
        Inputs:
            n: int
                delay between input and response
            length: int
                length of input / output array
            pct_signal: float
                percent of point which are signal points
        """
        if pct_signal > 1 or pct_signal < 0: raise ValueError("pct_signal has to be in [0,1]") 
        if n < 0: raise ValueError("n must be > 0")
        n_points_signal: int = int(length * pct_signal)
        indices_signals: np.ndarray = np.random.choice(length, size=n_points_signal, replace=False)
        
        x: np.ndarray = np.zeros(length)
        x[indices_signals] = 1
        y: np.ndarray = np.zeros(length)
        y[[k + n for k in indices_signals if k + n < length]] = 1

        return x, y




class ESNtest:

    def test1():
        """
        
        """
        x_train, y_train = DataLoader.delay_timeseries(n=5, length=10000, pct_signal=0.01)
        x_test, y_test = DataLoader.delay_timeseries(n=5, length=1000, pct_signal=0.01)

        esn = ESN(n_inputs = 1,
            n_outputs = 1,
            n_reservoir = 500,
            W_in_scaling = 0.2, # 0.8,
            teacher_forcing = False,
            is_SLM=False, #intensity=0.5418,
            leaky_rate = 0,
            wash_out = 50,
            noise = 0, # 5e-3,
            sparsity = 0.7, # Reservoir connectivity in the subject
            spectral_radius = 0.8,
            # learn_method='ridge', ridge_noise= 5 * 10 ** (-3),
            # learn_method="SGD", SGD_clf=clf,
            learn_method="pinv",
            random_state = 12,
            silent = False)
        
        pred_train = esn.fit(x_train, y_train)
        pred_test = esn.predict(x_test, continuation=False)
        n_points_plot = 200
        plt.plot(x_train, color="blue", label="input")
        plt.plot(y_train, color="green", label="real")
        plt.plot(pred_train, color="red", label="pred")
        plt.show()

        plt.plot(x_test, color="blue", label="input")
        plt.plot(y_test, color="green", label="real")
        plt.plot(pred_test, color="red", label="pred")
        plt.show()
        

    def test2():
        """"""
        x_train, y_train = DataLoader.periodic_signal(length=10000, period=5)
        x_test, y_test = DataLoader.periodic_signal(length=1000, period=5)
        esn = ESN(n_inputs = 1,
            n_outputs = 1,
            n_reservoir = 50,
            W_in_scaling = None, # 0.2,
            teacher_forcing = False,
            is_SLM=False, #intensity=0.5418,
            leaky_rate = 0, # 0.2
            wash_out = 0,
            noise = 5e-5,
            sparsity = 0.9, # Reservoir connectivity in the subject
            spectral_radius = 0.8,
            learn_method='ridge', ridge_noise= 5 * 10 ** (-3),
            # learn_method="SGD", SGD_clf=clf,
            # learn_method="pinv",
            random_state = 12,
            silent = False)
        pred_train = esn.fit(x_train, y_train)
        pred_test = esn.predict(x_test, continuation=False)
        plt.plot(pred_train[200:300], color="red", label="pred")
        plt.plot(y_train[200:300], color="green", label="real")
        plt.plot(x_train[200:300], color="blue", label="input")
        plt.show()

        plt.plot(pred_test[200:300], color="red", label="pred")
        plt.plot(y_test[200:300], color="green", label="real")
        plt.plot(x_test[200:300], color="blue", label="input")
        plt.show()

    def test3():
        """
        This is simple training test on MNIST
        """
        (x_train, y_train), (x_test, y_test) = DataLoader.load_mnist()
        clf = SGDClassifier(loss='modified_huber', penalty='l2', max_iter=2, epsilon=0.1,
                   class_weight='balanced', average=True)
        esn = ESN(n_inputs = 28 * 28,
            n_outputs = 10,
            n_reservoir = 500,
            W_in_scaling = 0.2,
            teacher_forcing = False,
            is_SLM=False, #intensity=0.5418,
            leaky_rate = 0,
            wash_out = 0,
            noise = 0, #5e-3,
            sparsity = 0.7, # Reservoir connectivity in the subject
            spectral_radius = 0.9,
            learn_method='ridge', ridge_noise=5*10**(-3),
            # learn_method="SGD", SGD_clf=clf,
            random_state = 12,
            silent = False)
        pred_train = esn.fit(x_train, y_train)
        pred_test = esn.predict(x_test, continuation=False)
        train_acc = accuracy(pred_train, y_train)
        test_acc = accuracy(pred_test, y_test)
        print(f"Training accuracy: {100*train_acc:.2f}%")
        print(f"Testing accuracy: {100*test_acc:.2f}%")
        pass


if __name__ == "__main__":
    ESNtest.test3()