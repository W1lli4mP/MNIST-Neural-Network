import numpy as np
from neural_network import NeuralNetwork, loss_function
from config import BATCH_SIZE, EPOCHS

# actual training loop for a batch
def batch_loop(neural_network: NeuralNetwork, n_train: int, train_X_shuffled: np.ndarray, train_y_oh_shuffled: np.ndarray) -> float:
    epoch_loss = 0.0
    for i in range(0, n_train, BATCH_SIZE):
        batch_X = train_X_shuffled[i: i + BATCH_SIZE]
        batch_y_oh = train_y_oh_shuffled[i: i + BATCH_SIZE]

        batch_size = batch_X.shape[0]

        # forward propagation -> calculate loss -> backward propagation -> gradient descent
        # forward propagation
        a = neural_network.forward(batch_X)

        # calculate loss
        loss = loss_function(a, batch_y_oh)
        epoch_loss += loss

        # backward propagation (includes gradient descent)
        neural_network.backward(batch_X, batch_y_oh, batch_size)
    
    # return new epoch loss
    return epoch_loss

def epoch_loop(neural_network: NeuralNetwork, train_X: np.ndarray, train_y_oh: np.ndarray, test_X: np.ndarray, test_y: np.ndarray):
    """
    trains and tests the neural network in epochs
    """
    n_train = train_X.shape[0]

    for epoch in range(1, EPOCHS):
        perm = np.random.permutation(n_train)
        train_X_shuffled = train_X[perm]
        train_y_oh_shuffled = train_y_oh[perm]

        epoch_loss = batch_loop(neural_network, n_train, train_X_shuffled, train_y_oh_shuffled)
        epoch_loss /= n_train

        # evaluate on test set
        test_a = neural_network.forward(test_X)
        preds = np.argmax(test_a, axis=1)
        test_acc = np.mean(preds == test_y)

        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {epoch_loss:.6f}  test_acc: {test_acc:.4f}")
