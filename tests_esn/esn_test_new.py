""""""
import sys
import os

from sklearn.linear_model import SGDClassifier
try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass
from new_pyESN import ESN
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


class DataCreator:
    def simple_sinus(length: int = 1000, f: float = 0.2):
        """
        Args:
            length (int): length of input/output signals
            f (float): frequency of the sinus
        Returns:
            input and output array for the test
        """
        x: np.ndarray = np.zeros(shape=(length, 1))
        y: np.ndarray = 0.5 * np.sin(f * np.linspace(1, length, length))
        return x, y

    def periodic_signal(length: int = 1000, period: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates an empty input signal, and an output signal of evenly spaced spikes.
        """
        x: np.ndarray = np.zeros((length, 1))
        y: np.ndarray = np.zeros((length, 1))
        y[[k for k in range(length) if k % period == 0]] = 1
        return x, y

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
    

class ESNtest:
    def test_sinus():
        """
        This tests is described here:
        https://sergiopeignier.github.io/teaching/python/leaky-echo-state.pdf

        The idea is to put a sinus signal as target and nothing as input and that the network only learns
        from the feedback.
        y_target(n) = 0.5 . sin(n . 0.2)
        x(n) = 0 for every n
        """
        n: int = 1000
        f: float = 0.2
        x, y = DataCreator.simple_sinus(length=1000, f=0.2)

        esn = ESN(
            n_inputs=1,
            n_outputs=1,
            n_reservoir=20,
            spectral_radius=0.8,
            sparsity=0.5,
            silent=False,
            input_scaling=0,
            feedback_scaling=1,
            leaky_rate=0,
            noise=0,
            wash_out=10
        )
        y_pred = esn.fit(inputs=x, outputs=y)
        mse = ((y - y_pred)**2).mean()
        plt.plot(y, label="real")
        plt.plot(y_pred, label="pred")
        plt.title("Sine wave generator ESN")
        plt.show()
        print(f"Loss: {mse:.2f}")
        return


    def test_periodic_signal():
        """
        The goal of this test is to measure the ESN capability to learn to predict a flat signal
        with only periodic spikes every N inputs.
        This is to test the memory capability of the ESN.
        """
        x, y = DataCreator.periodic_signal(length=1000, period=10)
        esn_args = {
            "n_inputs": 1, 
            "n_outputs": 1,
            "n_reservoir": 20,
            "spectral_radius": 0.8,
            "sparsity": 0.5,
            "silent": False,
            "input_scaling": 0,
            "feedback_scaling": 1,
            "leaky_rate": 0,
            "noise": 0,
            "wash_out": 100
        }
        esn = ESN(**esn_args)
        y_pred = esn.fit(x,y)
        mse = ((y - y_pred)**2).mean()
        plt.plot(y, label="real")
        plt.plot(y_pred, label="pred")
        plt.title("ESN on Periodic signal")
        plt.show()
        print(f"Loss: {mse:.2f}")

        """
        In this second part, we will now study what is the relation between the reservoir 
        size and the performance of the task.
        Globally we observe that after n_reservoir > period, the problem solving is satisfactory
        """
        del esn_args["n_reservoir"]
        esn_args["silent"] = True
        reservoir_size: list[int] = [*range(5, 40)]
        loss_array: np.ndarray = np.zeros(len(reservoir_size))
        for idx, k in enumerate(reservoir_size):
            esn = ESN(**esn_args, n_reservoir=k)
            y_pred = esn.fit(x, y)
            loss_array[idx] = ((y - y_pred)**2).mean()
        
        plt.plot(reservoir_size, loss_array, label="Loss")
        plt.xlabel("Reservoir size")
        plt.ylabel("Loss on periodic signal task")
        plt.title("Loss dependence on reservoir size on Periodic signal task")
        plt.show()
        return


if __name__ == "__main__":
    # ESNtest.test_sinus()
    ESNtest.test_periodic_signal()