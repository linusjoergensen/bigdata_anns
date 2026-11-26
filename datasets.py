import os
from urllib.request import urlretrieve
import traceback

import numpy as np
import h5py


SUPPORTED_DATASETS = ["sift-128-euclidean",
                      "mnist-784-euclidean",
                      "fashion-mnist-784-euclidean",
                      "glove-25-angular",
                      "glove-50-angular",
                      "glove-100-angular",
                      "glove-200-angular",
                      "deep-image-96-angular"]

def random_dataset(n: int, d: int, random_seed: int) -> np.ndarray:
    """
    Creates a random dataset.
    
    :param n: Number of vectors
    :type n: int
    :param d: Dimension of each vector
    :type d: int
    :param random_seed: Random seed to use for generation
    :type random_seed: int
    :return: Description
    :rtype: ndarray[_AnyShape, dtype[Any]]
    """
    raise NotImplementedError


def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join("data", f"{dataset_name}.hdf5")


def download(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.
    
    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(destination_path):
        print(f"Downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)

def get_dataset(dataset_name: str) -> tuple[h5py.File, int]:
    """
    Docstring for get_dataset
    
    :param dataset: Description
    :type dataset: str
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(f"{dataset_name} is not supported.")
    
    hdf5_filename = get_dataset_fn(dataset_name)
    try:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
        download(dataset_url, hdf5_filename)
    except:
        raise FileNotFoundError(f"Could not download file {dataset_url}")

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension