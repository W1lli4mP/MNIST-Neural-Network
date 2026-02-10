import numpy as np
from neural_network import NeuralNetwork, loss_function
from config import BATCH_SIZE, EPOCHS

# actual training loop for a batch
def batch_loop(neural_network: NeuralNetwork, n_train: int, train_X_shuffled: np.ndarray, train_y_oh_shuffled: np.ndarray) -> float:
    """
    train all batches in an epoch (forward -> loss -> backward -> gradient descent) and return epoch loss
    """
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

def train_epoch(neural_network: NeuralNetwork, train_X: np.ndarray, train_y_oh: np.ndarray) -> float:
    """
    train for one epoch and return the loss
    """
    n_train = train_X.shape[0]
    perm = np.random.permutation(n_train)
    train_X_shuffled = train_X[perm]
    train_y_oh_shuffled = train_y_oh[perm]

    epoch_loss = batch_loop(neural_network, n_train, train_X_shuffled, train_y_oh_shuffled)
    return epoch_loss / n_train

def evaluate(neural_network: NeuralNetwork, test_X: np.ndarray, test_y: np.ndarray) -> float:
    """
    test the model and return the accuracy
    """
    test_a = neural_network.forward(test_X)
    preds = np.argmax(test_a, axis=1)
    accuracy = np.mean(preds == test_y)
    return accuracy

def train_loop(neural_network: NeuralNetwork, train_X: np.ndarray, train_y_oh: np.ndarray, test_X: np.ndarray, test_y: np.ndarray) -> None:
    """
    actual training/epoch loop with evaluation
    """
    for epoch in range(1, EPOCHS):
        loss = train_epoch(neural_network, train_X, train_y_oh)
        acc = evaluate(neural_network, test_X, test_y)
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {loss:.6f}  test_acc: {acc:.4f}")