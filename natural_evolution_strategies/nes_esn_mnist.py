"""
ESN x NES implementation example.
"""
import sys
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier

try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass
from pyESN import ESN
from utils import split_set, MnistDataloader, accuracy, pinv
from natural_evolution_strategies.NES import NES

INPUT_PATH = 'data'
TRAINING_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
TRAINING_LABELS_FILEPATH = os.path.join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
TEST_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
TEST_LABELS_FILEPATH = os.path.join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

def load_mnist():
    """Loads the MNIST dataset"""
    input_path = 'data'
    training_images_filepath = os.path.join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = os.path.join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = os.path.join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = os.path.join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(normalize=True)
    return (x_train, y_train), (x_test, y_test)


def test1_one_output():
    """
    This test program will fit a simple ESN on MNIST but instead of using SGD / PINV to 
    fit the output layer, we will use a Natural Evolution Strategy
    """
    mnist_dataloader = MnistDataloader(
        TRAINING_IMAGES_FILEPATH, TRAINING_LABELS_FILEPATH, 
        TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
        normalize=True, 
        # crop_top=2, crop_bot=2, crop_left=2, crop_right=2,
        # out_format="column",
        # hog={"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": True},
        projection=200,
        silent=False,
    )
    y_train = y_train[:, 2]
    y_test = y_test[:, 2]

    def custom_method(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """"""
        # We start from an array fitted on 100 samples
        # w: np.ndarray = pinv(x[1000:1100], y[1000:1100])
        w = np.zeros_like(w)
        def f_reward(w_temp: np.ndarray) -> float:
            return -np.linalg.norm(np.dot(x, w_temp.T) - y) / (y.shape[0] * y.shape[1])

        nes = NES(
            w=w,
            f=f_reward,
            pop=15,
            sigma=0.000005,
            alpha=0.01,
            mirrored_sampling=True
        )

        nes.optimize(n_iter=50, graph=False)
        
        return w
    
    n_samples, input_size = x_train.shape
    esn = ESN(
        n_inputs=input_size,
        n_outputs=1,
        spectral_radius=0.8,
        n_reservoir=20,
        sparsity=0.5,
        silent=False,
        input_scaling=0.7,
        feedback_scaling=0.2,
        wash_out=25,
        learn_method="custom",
        custom_method=custom_method,
    )
    pred_train = esn.fit(x_train, y_train)
    pred_test = esn.predict(x_test, continuation=False)
    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)
    print(f"Training accuracy: {100*train_acc:.2f}%")
    print(f"Testing accuracy: {100*test_acc:.2f}%")
    pred_test = esn.predict(x_test, continuation=False)
    final_loss = -np.linalg.norm(pred_test - y_test) / (y_test.shape[0] * y_test.shape[1])
    print(f"Loss: {final_loss}")
    esn.learn_method = "pinv"
    esn.fit(x_train, y_train)
    pred_test = esn.predict(x_test, continuation=False)
    loss_real = -np.linalg.norm(pred_test - y_test) / (y_test.shape[0] * y_test.shape[1])
    print(f"Real Loss: {loss_real}")
    return


def test1():
    """
    This test program will fit a simple ESN on MNIST but instead of using SGD / PINV to 
    fit the output layer, we will use a Natural Evolution Strategy
    """
    mnist_dataloader = MnistDataloader(
        TRAINING_IMAGES_FILEPATH, TRAINING_LABELS_FILEPATH, 
        TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
        normalize=True, 
        # crop_top=2, crop_bot=2, crop_left=2, crop_right=2,
        # out_format="column",
        # hog={"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": True},
        projection=200,
        silent=False,
    )

    def custom_method(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """"""
        # We start from an array fitted on 100 samples
        # w: np.ndarray = pinv(x[1000:1100], y[1000:1100])
        w = np.zeros_like(w)
        def f_reward(w_temp: np.ndarray) -> float:
            return -np.linalg.norm(np.dot(x, w_temp.T) - y) / (y.shape[0] * y.shape[1])

        nes = NES(
            w=w,
            f=f_reward,
            pop=15,
            sigma=0.000005,
            alpha=0.01,
            mirrored_sampling=True
        )
        for _ in range(10):
            nes.optimize(n_iter=50, graph=False)
            nes.alpha /= 2
        
        return w
    
    n_samples, input_size = x_train.shape
    esn = ESN(
        n_inputs=input_size,
        n_outputs=10,
        spectral_radius=0.8,
        n_reservoir=20,
        sparsity=0.5,
        silent=False,
        input_scaling=0.7,
        feedback_scaling=0.2,
        wash_out=25,
        learn_method="custom",
        custom_method=custom_method,
    )
    pred_train = esn.fit(x_train, y_train)
    pred_test = esn.predict(x_test, continuation=False)
    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)
    print(f"Training accuracy: {100*train_acc:.2f}%")
    print(f"Testing accuracy: {100*test_acc:.2f}%")
    pred_test = esn.predict(x_test, continuation=False)
    final_loss = -np.linalg.norm(pred_test - y_test) / (y_test.shape[0] * y_test.shape[1])
    print(f"Loss: {final_loss}")
    esn.learn_method = "pinv"
    esn.fit(x_train, y_train)
    pred_test = esn.predict(x_test, continuation=False)
    loss_real = -np.linalg.norm(pred_test - y_test) / (y_test.shape[0] * y_test.shape[1])
    print(f"Real Loss: {loss_real}")
    return


def test2():
    """
    This is an attempt to fit the input layer of the ESN with NES
    """
    (x_train, y_train), (x_test, y_test) = load_mnist()
    esn = ESN(
        n_inputs=28*28,
        n_outputs=10,
        spectral_radius=0.8,
        n_reservoir=20,
        sparsity=0.5,
        silent=False,
        input_scaling=0.7,
        feedback_scaling=0.2,
        wash_out=25,
        learn_method="sgd",
        learning_rate=0.00001
    )
    # def f_reward(x: np.ndarray) -> float:
    #     esn_temp = ESN(**esn_args)
    #     esn_temp.W_in = x
    #     pred_train = esn.fit(x_test, y_test)
    #     return accuracy(pred_train, y_test)
    w_base = esn.W_in
    def f_reward(x: np.ndarray) -> float:
        esn.W_in = x
        pred_test = esn.predict(x_test)
        return mean_squared_error(y_test, pred_test)

    esn.fit(x_train[:5000], y_train[:5000])
    esn.silent = True

    nes.optimize(n_iter=20, silent=False)


if __name__ == "__main__":
    test1()

