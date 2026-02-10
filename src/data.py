import numpy as np
from mnist import MNIST
from paths import here

def get_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    extract MNIST data and return data as a tuple of np arrays
    """
    data_path = here("data")
    mndata = MNIST(data_path)

    images, labels = mndata.load_training()
    train_X = np.array(images)
    train_y = np.array(labels)

    images, labels = mndata.load_testing()
    test_X = np.array(images)
    test_y = np.array(labels)

    return train_X, train_y, test_X, test_y

# one hot encoding for y (due to mismatched size)
def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    convert integer class labels into one-hot encoded vectors so the index of the correct class is 1 and all others are 0
    return one-hot encoded y
    """
    y = y.astype(int) # makes sure all data is int
    out = np.zeros((y.shape[0], num_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out