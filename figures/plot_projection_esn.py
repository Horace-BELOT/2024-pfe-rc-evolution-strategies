"""
This file contains the code that will show how the accuracy of the ESN evolves
with the dimension of the input state with regards to random projection to a 
lower dimensional space.
"""
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
from sklearn.decomposition import PCA
import umap
import numpy as np

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
    df.sort_values(by=["tag", "dim"], inplace=True)
    for tag in list(df.loc[:, "tag"].unique()):
        # creating mask
        m = df["tag"] == tag
        plt.plot(df.loc[m, "dim"], df.loc[m, "test_accuracy"], 
                label=tag, **marker_kwargs)
    plt.ylabel("Accuracy")
    plt.xlabel("Dimension")
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path is not None: plt.savefig(save_path)
    plt.show()


def main(
        df_path: str = "figures/plot_projection_df.csv",
        figure_path: str = "figures/plot_projection_graph.png",
        tag: str = "normal"
):
    """"""
    mnist_dataloader = MnistDataloader(TRAINING_IMAGES_FILEPATH, TRAINING_LABELS_FILEPATH,
                                        TEST_IMAGES_FILEPATH, TEST_LABELS_FILEPATH)
    
    proj_dimensions: List[int] = [
        1,2,3,4,
        *range(5, 50, 5),
        *range(50, 100, 10),
        *range(100, 151, 25),
        *range(200, 300, 50),
        *range(300, 501, 100),
        #600, 700, 784, 912
    ]
    print(proj_dimensions)
    result_records: List[Dict[str, Any]] = []
    for input_dim in tqdm.tqdm(proj_dimensions):
        x_train, y_train, x_test, y_test = None, None, None, None
        if tag == "hog":
            (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
                normalize=True, 
                # crop_top=2, crop_bot=2, crop_left=2, crop_right=2,
                # out_format="column",
                hog={"image_shape": (28,28), "cell": (8,8), "block": (2,2), "keep_inputs": False},
                projection=100,
                silent=True,
            )
        elif tag == "normal":
            (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
                normalize=True, 
                projection=input_dim,
                silent=True,
            )
        else:
            (x_train, y_train), (x_test, y_test) = mnist_dataloader.prepare_data(
                normalize=True, 
                silent=True,
            )
        # do a PCA projection on the data
        if tag == "pca":
            pca = PCA(n_components=input_dim)
            x_train = pca.fit_transform(x_train)
            x_test = pca.transform(x_test)
        
        # do a UMAP
        if tag == "umap":
            reducer = umap.UMAP(n_components=input_dim)
            x_train = reducer.fit_transform(x_train)
            x_test = reducer.transform(x_test)
        
        # display the explained variance ratio with the number of components
        # print("Explained Variance Ratio of the components", np.sum(pca.explained_variance_ratio_))
        
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
            "tag": tag,
        })

        # Saving data by concatenating with what we already have and keeping the latest
        df = pd.read_csv(df_path, sep=";")
        df2: pd.DataFrame = pd.DataFrame.from_records(result_records)
        df = pd.concat([df, df2])
        df.sort_values(by=["test_accuracy"], ascending=True, inplace=True)
        df.drop_duplicates(subset=["dim", "tag"], keep="last", inplace=True)
        df.sort_values(by=["tag", "dim"], inplace=True)
        df.to_csv(df_path, index=False, sep=";")

    # Updating plot
    plot_data(df, figure_path)

if __name__ == "__main__":
    # main(tag="normal")
    # main(tag="hog")
    # main(tag="pca")
    # main(tag="umap")
    df = pd.read_csv("figures/plot_projection_df.csv", sep=";")
    plot_data(df, save_path="figures/plot_projection_graph.png")