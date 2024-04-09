""""""
import sys
import os

try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict

from natural_evolution_strategies.NES import NES
from pyESN import ESN
from utils import sgd, MnistDataloader, accuracy, pinv

input_path = 'data'
training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

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

    def periodic_signal(length: int = 1000, period: int = 5) -> Tuple[np.ndarray, np.ndarray]:
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
            wash_out=10,
            random_state=13  # random number
        )
        y_pred = esn.fit(inputs=x, outputs=y)
        mse = ((y - y_pred)**2).mean()
        plt.plot(y, label="real")
        plt.plot(y_pred, label="pred")
        plt.title("Sine wave generator ESN")
        plt.show()
        print(f"Loss: {mse:.2f}")
        return


    def periodic_signal():
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
        reservoir_size: List[int] = [*range(5, 40)]
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
    
    def mnist_simple():
        # Set file paths based on added MNIST Datasets
        input_path = 'data'
        training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
        # Load MINST dataset
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
            normalize=True, 
            # crop_top=2, crop_bot=2, crop_left=2, crop_right=2,
            # out_format="column",
            # hog={"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": True},
            projection=100,
            silent=False,
        )
        n_samples, input_size = x_train.shape
        esn = ESN(
            n_inputs=input_size,
            n_outputs=10,
            spectral_radius=0.9,
            n_reservoir=500,
            sparsity=0.7,
            silent=False,
            input_scaling=0.7,
            feedback_scaling=0.2,
            leaky_rate=0.7,
            wash_out=25,
            learn_method="pinv",
            learning_rate=0.00001,
            allow_cut_connections=True,
            repeated_inputs=5,
        )
        pred_train = esn.fit(x_train, y_train)
        pred_test = esn.predict(x_test, continuation=False)
        train_acc = accuracy(pred_train, y_train)
        test_acc = accuracy(pred_test, y_test)
        
        print(f"Training accuracy: {100*train_acc:.2f}%")
        print(f"Testing accuracy: {100*test_acc:.2f}%")
        return
    
    def linear_reg_mnist():
        """
        This is NOT AN ESN test
        This is just to see how well a simple linear regression performs on MNIST
        The result are: 
        "
            Result of a simple linear regression on MNIST data
            Training accuracy: 84.58%
            Testing accuracy: 85.31%
        "
        """
        # Set file paths based on added MNIST Datasets
        input_path = 'data'
        training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
        # Load MINST dataset
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
            normalize=True,
            hog={"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": True},
            # projection=100,
            silent=False,
        )
        # theta = sgd(x_train, y_train, alpha=5*10**-6, silent=False, lambda_ridge=10**-5)
        theta = pinv(x_train, y_train)
        pred_train = np.dot(x_train, theta.T)
        pred_test = np.dot(x_test, theta.T)
        train_acc = accuracy(pred_train, y_train)
        test_acc = accuracy(pred_test, y_test)
        print("Result of a simple linear regression on MNIST data")
        print(f"Training accuracy: {100 * train_acc:.2f}%")
        print(f"Testing accuracy: {100 * test_acc:.2f}%")
        return
    
    def training_output_layer_nes_vs_sgd() -> None:
        """
        In this test, we will compare the training of the output layer between 2 methods:
         1 - training with NES
         2 - training with SGD
        """
        # Load MINST dataset
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
            normalize=True, 
            crop_top=2, 
            crop_bot=2, 
            crop_left=2, 
            crop_right=2,
            # out_format="column"
        )
        truncate: int = 5000
        x_train = x_train[:truncate, :]
        y_train = y_train[:truncate, :]
        n: int = x_train.shape[0]  # number of samples as training inputs

        # We start by defining the custom method using the NES

        
        nes = NES(
            w=W_out_nes,
            f=f_reward,
            pop=25,
            sigma=0.0005,
            alpha=0.003,
            f_test=f_test,
        )




        esn = ESN(
            n_inputs=24 * 24,  # truncated by 2 on each side
            n_outputs=10,
            spectral_radius=0.8,
            n_reservoir=500,
            sparsity=0.5,
            silent=False,
            input_scaling=0.7,
            feedback_scaling=0.2,
            wash_out=25,
            learn_method="pinv",
            learning_rate=0.00001
        )
        
        # Feed the inputs weights to the network
        x: np.ndarray = esn.feed(inputs=x_train, outputs=y_train, wash_out=esn.wash_out)[0].copy()
        y: np.ndarray = y_train[esn.wash_out:, :].copy()
        # We will play around with W_out a lot so we save a backup right away
        W_out_nes: np.ndarray = esn.W_out.copy()
        W_out_sgd: np.ndarray = esn.W_out.copy()

        # Defining the NES variables
        # We will feed the testing samples to the network only once and ahead of the NES iterations
        # This will prevent us from having to refeed the whole test by calling esn.test() every time
        x_extend_test, _ = esn.feed(inputs=x_test, outputs=None, wash_out=1)
        y_extend_test = y_test[1:,:]
        
        def f_reward(w_out_i: np.ndarray) -> float:
            """
            Evaluates the loss on the train set with W_out = w_out_i
            Inputs: 
                w_out_i: np.ndarray
            """
            # pred_test = esn.predict(x_test)
            n_train = x.shape[0]
            # train_pred = np.zeros((n_train, 10))
            # for k in range(n_train - 1):
            #     train_pred[k + 1, :] = np.dot(w_out_i, x[k + 1])
            # Compares the prediction with the training labels y_train
            # return - mean_squared_error(y, train_pred)
            return -np.linalg.norm(np.dot(x, w_out_i.T) - y) / (y.shape[0] * y.shape[1])
        
        def f_test(w: np.ndarray) -> float:
            """Evaluates the testing loss with W_out = w"""
            return -np.linalg.norm(
                np.dot(x_extend_test, w.T) - y_extend_test) / (y_extend_test.shape[0] * y_extend_test.shape[1])


        training_loss = nes.optimize(n_iter=100, silent=False)
        plt.plot(np.log10(-training_loss), label="Training loss")
        plt.plot(np.log10(-nes.testing_loss), label="Testing loss")
        pred_test = esn.predict(x_test)
        print(f"Current accuracy: {accuracy(pred_test, y_test)}")
        print(f"Current loss: {np.linalg.norm(pred_test - y_test)}")
        plt.show()
        return
    
    def comparison():
        """"""
        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
            normalize=True, 
            crop_top=2, 
            crop_bot=2, 
            crop_left=2, 
            crop_right=2,
            out_format="column",
            # projection=100,
        )
        truncate: int = 5000
        x_train = x_train[:truncate, :]
        y_train = y_train[:truncate, :]
        n_inputs: int = x_train.shape[1]
        n_outputs: int = y_train.shape[1]
        itor: int = mnist_dataloader.input_to_output_ratio
        esn = ESN(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            spectral_radius=0.8,
            n_reservoir=500,
            sparsity=0.5,
            silent=False,
            input_scaling=0.7,
            feedback_scaling=0.2,
            wash_out=25,
            learn_method="pinv",
            learning_rate=0.00001,
            input_to_output_ratio=itor,
        )
        pred_train = esn.fit(x_train, y_train)
        pred_test = esn.predict(x_test, continuation=False)
        train_acc = accuracy(pred_train, y_train)
        test_acc = accuracy(pred_test, y_test)
        print(f"Training accuracy: {100*train_acc:.2f}%")
        print(f"Testing accuracy: {100*test_acc:.2f}%")
        return


if __name__ == "__main__":
    # ESNtest.linear_reg_mnist()
    # ESNtest.test_sinus()
    # ESNtest.periodic_signal()
    ESNtest.mnist_simple()
    # ESNtest.training_output_layer_nes_vs_sgd()
    # ESNtest.comparison()