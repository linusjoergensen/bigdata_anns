import numpy as np

SUPPORTED_DATASETS = []

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

def get_dataset(dataset: str):
    """
    Docstring for get_dataset
    
    :param dataset: Description
    :type dataset: str
    """
    raise NotImplementedError

print("I did everything very cool")