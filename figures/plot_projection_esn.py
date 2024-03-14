"""
This file contains the code that will show how the accuracy of the ESN evolves
with the dimension of the input state with regards to random projection to a 
lower dimensional space.
"""
import multiprocessing
import os
import time
import sys

try:  # Fixing import problems
    if "pyESN.py" not in os.listdir(sys.path[0]):
        upper_folder: str = "\\".join(sys.path[0].split("\\")[:-1])
        sys.path.append(upper_folder)
except:
    pass

import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from typing import List, Dict, Any, Optional

from utils import accuracy, MnistDataloader
from pyESN import ESN


INPUT_PATH = 'data'
TRAINING_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
TRAINING_LABELS_FILEPATH = os.path.join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
TEST_IMAGES_FILEPATH = os.path.join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
TEST_LABELS_FILEPATH = os.path.join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

def plot_data(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """"""
    marker_kwargs: Dict[str, Any] = {
        "marker": "o",
        "markersize": 5
    }
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(df.loc[:, "dim"], df.loc[:, "train_accuracy"], 
             label="Train accuracy", **marker_kwargs)
    ax2.plot(df.loc[:, "dim"], df.loc[:, "test_accuracy"], 
             label="Test accuracy", **marker_kwargs)

    plt.subplots_adjust(hspace=0.2)
    if save_path is not None: plt.savefig(save_path)
    plt.show()


def main(
        df_path: str = "figures/plot_projection_df.csv",
        figure_path: str = "figures/plot_projection_graph.png",
):
    """"""
    mnist_dataloader = MnistDataloader(TRAINING_IMAGES_FILEPATH, TRAINING_LABELS_FILEPATH,
                                        TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH)
    
    proj_dimensions: List[int] = [
        *range(10, 50, 10),
        #*range(100, 300, 50),
        #*range(300, 501, 100)
    ]
    result_records: List[Dict[str, Any]] = []
    for input_dim in tqdm.tqdm(proj_dimensions):
        (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
            normalize=True, 
            # crop_top=2, crop_bot=2, crop_left=2, crop_right=2,
            # out_format="column",
            # hog={"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": True},
            projection=input_dim,
            silent=True,
        )
        t_start: float = time.time()
        esn = ESN(
            n_inputs=input_dim,
            n_outputs=10,
            spectral_radius=0.9,
            n_reservoir=500,
            sparsity=0.7,
            silent=True,
            input_scaling=0.7,
            feedback_scaling=0.2,
            leaky_rate=0.7,
            wash_out=25,
            learn_method="pinv",
            learning_rate=0.00001
        )
        pred_train = esn.fit(x_train, y_train)
        pred_test = esn.predict(x_test, continuation=False)
        train_acc = accuracy(pred_train, y_train)
        test_acc = accuracy(pred_test, y_test)
        computation_time: float = time.time() - t_start
        print(f"\n -- Dimension {input_dim} done in {computation_time:.2f} s")
        print(f"Training accuracy: {100 * train_acc:.2f}%")
        print(f"Testing accuracy: {100 * test_acc:.2f}%")

        result_records.append({
            "dim": input_dim,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "time": computation_time,
        })

    # Saving data
    df: pd.DataFrame = pd.DataFrame.from_records(result_records)
    df.to_csv(df_path, index=False, sep=";")

    # Plotting
    plot_data(df, figure_path)

if __name__ == "__main__":
    # main()
    df = pd.read_csv("figures/plot_projection_df.csv", sep=";")
    plot_data(df)