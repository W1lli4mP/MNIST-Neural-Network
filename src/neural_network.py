from paths import here
from mnist import MNIST
import numpy as np

def get_data():
    data_path = here("data")
    mndata = MNIST(data_path)

    images, labels = mndata.load_training()
    train_X = np.array(images)
    train_y = np.array(labels)

    images, labels = mndata.load_testing()
    test_X = np.array(images)
    test_y = np.array(labels)

    return train_X, train_y, test_X, test_y

# make NN a class??
# weights
# biases
# activation (needs function)
# layers
# forward/backward propogation
# loss function (how wrong the model was at a data point)
# (a - y)^2
# cost function (how wrong the model was overall)
# 1/N * sum(a - y)^2
# activation function (ReLU)