import numpy as np
from neural_network import NeuralNetwork
from data import get_data, one_hot
from train import epoch_loop

def main():
    # INITIALISE NN
    nn = NeuralNetwork([784, 128, 10])

    # EXTRACTING THE DATA
    train_X, train_y, test_X, test_y = get_data()

    # preprocessing for inputs: making them between 0-1 using 255 (MNIST uses grayscale)
    train_X = train_X.astype(np.float32) / 255
    test_X = test_X.astype(np.float32) / 255
    train_y_oh = one_hot(train_y, 10)

    # start training NN
    print("STARTING")
    epoch_loop(nn, train_X, train_y_oh, test_X, test_y)

    # save model
    # save_model(nn, "prototype1.npz")

main()