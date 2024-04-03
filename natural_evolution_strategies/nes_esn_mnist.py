"""
ESN x NES implementation example.
"""
import sys
import os
import multiprocessing
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDClassifier
from typing import Dict, List, Any, Optional

try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass
from pyESN import ESN
from utils import split_set, MnistDataloader, accuracy, pinv, save_pickle, load_pickle
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
    (x_train, y_train_base), (x_test, y_test_base) = mnist_dataloader.prepare_data(
        normalize=True, 
        # crop_top=2, crop_bot=2, crop_left=2, crop_right=2,
        # out_format="column",
        # hog={"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": True},
        projection=200,
        silent=False,
    )
    out_matrixes = []
    for k in range(10):
        print(f"\n\nCurrent index: {k}")
        y_train = y_train_base[:, k].reshape((y_train_base.shape[0], 1))
        y_test = y_test_base[:, k].reshape((y_test_base.shape[0], 1))

        def custom_method(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            """"""
            # We start from an array fitted on 100 samples
            w: np.ndarray = pinv(x[1000:1100], y[1000:1100])
            w = np.zeros_like(w)
            def f_reward(w_temp: np.ndarray) -> float:
                return -np.linalg.norm(np.dot(x, w_temp.T) - y) / (y.shape[0] * y.shape[1])

            nes = NES(
                w=w,
                f=f_reward,
                pop=25,
                sigma=5 * 10 ** (-10),
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
            n_reservoir=50,
            sparsity=0.5,
            silent=False,
            input_scaling=0.7,
            feedback_scaling=0.2,
            wash_out=25,
            learn_method="custom",
            custom_method=custom_method,
            random_state=12
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
        w: np.ndarray = pinv(x[1000:1100], y[1000:1100])
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

    # nes.optimize(n_iter=20, silent=False)


def function_for_each_process(i: int):
    """
    This is the function that will be executed in parallel in each process in the
    next function (parallel_run_mnist).

    Args:
        i: int
            which target [0, 9] of MNIST to regress using NES
    """
    print(f"Process {i}: Starting")
    esn: ESN = ESN.load("saved_data/base_esn.pickle")

    # We define the start of the NES object
    # NES object will be updated inside custom_method
    nes = NES(
        w=esn.W_out,
        f=lambda x: None,  # This function will be changed after
        pop=15,
        sigma=0.000005,
        alpha=0.01,
        mirrored_sampling=True,
        adaptive_rate=True,
    )
    
    def custom_method(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """"""
        # We start from an array fitted on 100 samples
        w = esn.W_out.copy()
        def f_reward(w_temp: np.ndarray) -> float:
            return -np.linalg.norm(np.dot(x, w_temp.T) - y) / (y.shape[0] * y.shape[1])
        nes.f = f_reward
        nes.w = w

        nes.optimize(n_iter=2000, graph=False)
        
        return w
    
    # We change the learning method
    esn.learn_method = "custom"
    esn.custom_method = custom_method

    # We load the mnist data
    mnist_dict: Dict[str, np.ndarray] = load_pickle("saved_data/mnist_data_test_parallel_run.pickle")
    x_train = mnist_dict["x_train"]
    y_train = mnist_dict["y_train"]
    x_test = mnist_dict["x_test"]
    y_test = mnist_dict["y_test"]
    input_size: int = x_train.shape[1]

    # We only keep 1 output
    y_train = y_train[:, i].reshape((y_train.shape[0], 1))
    y_test = y_test[:, i].reshape((y_test.shape[0], 1))

    print(f"Process {i}: Training")
    # print(f"Mean of weights pre training: {np.abs(esn.W_out).mean()}")
    pred_train = esn.fit(x_train, y_train)
    pred_test = esn.predict(x_test, continuation=False)
    train_acc = accuracy(pred_train, y_train)
    test_acc = accuracy(pred_test, y_test)
    # print(f"Mean of weights post training: {np.abs(esn.W_out).mean()}")
    # print(f"Sum of preds: {pred_test.sum()}")
    # print((np.hstack([pred_test, y_test])[:25,:]).T)
    print(f"Training accuracy for process {i}: {100 * train_acc:.2f}%\n" + 
          f"Testing accuracy for process {i}: {100 * test_acc:.2f}%\n" +
          f"Process {i} finished !\n")

    # We then save the model:
    path_i: str = f"saved_data/results_{i}.pickle"
    print(f"Process {i}: Saving")
    save_pickle(
        file_path=path_i,
        obj={
            "i": i,
            "df": nes.data,
            "W_out": esn.W_out
        }
    )
    print(f"Process {i}: Done")
    return (1,1)


def plot_results_parallel_run(
        df: Optional[pd.DataFrame],
        save_path: str,
    ) -> None:
    """
    Plots the result of the accuracy
    """
    if isinstance(df, str):
        df = pd.read_csv(df, sep=";")

    # fig, axs = plt.subplots(2)
    for target_value in df["target_mnist"].unique():
        subdf = df.loc[df["target_mnist"] == target_value, :].copy()
        subdf = subdf.sort_values(["index"], ascending=True)
        plt.plot(subdf["index"], subdf["train_loss"], label=f"Number {target_value}")
    plt.xscale("log")
    plt.savefig(save_path)
    plt.show()

    

def parallel_run_mnist():
    """
    Trains ESN's output layer on MNIST by splitting the 10 classes of the MNIST dataset (10 classes)
    and training the 10 parts of the output layer separetely on different cores using Natural Evolution
    Strategies.
    """
    # Loads MNIST data
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
    n_inputs: int = x_train.shape[1]

    # We then create an ESN with only 1 output
    esn = ESN(
        n_inputs=n_inputs,
        n_outputs=1,
        spectral_radius=0.8,
        n_reservoir=50,
        sparsity=0.5,
        silent=True,
        input_scaling=0.7,
        feedback_scaling=0.2,
        wash_out=25,
        learn_method="sgd",
        learning_rate=0.00001
    )
    # We then save the model
    esn.save(file_path="saved_data/base_esn.pickle")
    # We save the data (because the projection is sometimes not reproductible)
    save_pickle({
        "x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test,
        }, file_path="saved_data/mnist_data_test_parallel_run.pickle"
    )
    
    # We delete some objects to free memory
    del esn; del x_train; del y_train; del x_test; del y_test
    with multiprocessing.Pool(processes=10) as pool:
        for i in range(10):
            pool.apply_async(function_for_each_process, (i,))
        pool.close()
        pool.join()
    print(f"\n\nFinished the multiple processes\n\n")
    results = {i: load_pickle(f"saved_data/results_{i}.pickle")
               for i in range(10)}
    
    # We load the mnist data
    mnist_dict: Dict[str, np.ndarray] = load_pickle("saved_data/mnist_data_test_parallel_run.pickle")
    x_train = mnist_dict["x_train"]
    y_train = mnist_dict["y_train"]
    x_test = mnist_dict["x_test"]
    y_test = mnist_dict["y_test"]

    # We rebuild the ESN object
    esn: ESN = ESN.load("saved_data/base_esn.pickle")
    # We rebuild the entire array
    esn.W_out = np.vstack([
        results[i]["W_out"] for i in range(10)
    ])
    # We then need to change the number of outputs
    esn.n_outputs = 10
    # Feedback loop shape doesnt match anymore
    esn.W_fb = esn.random_state_.rand(esn.n_reservoir, esn.n_outputs) * 2 - 1
    esn.W_fb *= esn.feedback_scaling
    esn.silent = False
    pred_test = esn.predict(x_test, continuation=False)
    test_acc = accuracy(pred_test, y_test)
    print(f"Final Testing accuracy: {100*test_acc:.2f}%")
    esn.save("saved_data/final_esn_trained_with_nes.pickle")
    for i in range(10):
        results[i]["df"]["target_mnist"] = i
        results[i]["df"]["index"] = results[i]["df"].index
    df = pd.concat([results[i]["df"] for i in range(10)])
    df.to_csv("saved_data/loss_data_df.csv", sep=";", index=True)


def main():
    # test1_one_output()
    # parallel_run_mnist()
    
    # Parallel run = NES training output layer 
    plot_results_parallel_run(
        df="saved_data/loss_data_df.csv",
        save_path="saved_data/train_graph.png"
    )


if __name__ == "__main__":
    main()