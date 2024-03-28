from cmaes import CMA
import os
import sys
try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass

from pyESN import ESN
from utils import split_set, MnistDataloader, accuracy, pinv, save_pickle, load_pickle
import umap
import numpy as np


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

def mnist_reduced(input_dim=4):
    (x_train, y_train), (x_test, y_test) = load_mnist()
    reducer = umap.UMAP(n_components=input_dim)
    x_train = reducer.fit_transform(x_train)
    x_test = reducer.transform(x_test)
    return (x_train, y_train), (x_test, y_test)

def train_esn(input_size,reservoir_size=50):
    (x_train, y_train), (x_test, y_test) = mnist_reduced(input_size)
    esn = ESN(
        n_inputs=input_size,
        n_outputs=10,
        spectral_radius=0.8,
        n_reservoir=reservoir_size,
        sparsity=0.5,
        silent=False,
        input_scaling=0.7,
        feedback_scaling=0.2,
        wash_out=25,
        learn_method="pinv",
        random_state=12
    )
    
    upper_bound = np.ones(esn.W_in.shape[0]*esn.W_in.shape[1])
    lower_bound = -np.ones(esn.W_in.shape[0]*esn.W_in.shape[1])
    bounds = np.stack((lower_bound, upper_bound), axis=1)
    optimizer = CMA(mean=np.zeros(esn.W_in.shape[0]*esn.W_in.shape[1]), sigma=1, bounds=bounds, population_size=50)
    best_sol = None
    best_error = np.inf
    try:
        for generation in range(100):
            solutions = []
            for i in range(optimizer.population_size):
                esn_result = optimizer.ask()
                esn.W_in = esn_result.reshape(esn.W_in.shape)
                pred_train = esn.fit(x_train, y_train)
                pred_test = esn.predict(x_test, continuation=False)
                train_acc = accuracy(pred_train, y_train)
                test_acc = accuracy(pred_test, y_test)
                error = 1 - test_acc
                if error < best_error:
                    best_error = error
                    best_sol = esn.W_in
                print(error)
                solutions.append((esn_result, error))
            print("Generation: ", generation, "Best error: ", np.min([x[1] for x in solutions]))
            optimizer.tell(solutions)
    except KeyboardInterrupt:
        pass
    esn.W_in = best_sol
    print("After all the generations, the best error is: ", best_error)

if __name__ == "__main__":
    train_esn(4, 50)