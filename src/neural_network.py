from paths import here
from mnist import MNIST
import numpy as np

LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100

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

# one hot encoding for y (due to mismatched size)
def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = y.astype(int) # makes sure all data is int
    out = np.zeros((y.shape[0], num_classes), dtype=float)
    out[np.arange(y.shape[0]), y] = 1.0
    return out

# activation functions
def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

# NOTE: hold back on softmax as the output activation (does not pair well with ReLU)
# TODO: implement cross-entrophy loss function to pair with softmax
"""
# activation function for output layer only
def softmax(z: np.ndarray) ->np.ndarray:
    # softmax depends on relative differences, so reduce the number size by subtracting by the maximum to avoid stuff like (inf/ inf)
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z) # e^z
    sum = np.sum(ez, axis=-1, keepdims=True) # axis = -1 sums up all of the classes to 1 and not anything else
    return ez / sum

def softmax_derivative(z: np.ndarray) -> np.ndarray:
    # TODO: combine softmax with upstream gradient to calculate this and simplify jacobian matrix
    pass
"""

# activation function derivatives
def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def loss_function(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((a - y) ** 2)

def loss_function_derivative(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (a - y) # with respect to a
class NeuralNetwork:
    def __init__(self):
        pass

class Layer:
    def __init__(self, biases: np.ndarray, weights: np.ndarray) -> None:
        self.b = biases
        self.W = weights

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return ReLU(z)

    def activation_function_derivative(self, z: np.ndarray) -> np.ndarray:
        return ReLU_derivative(z)

    def linearity(self, x: np.ndarray) -> np.ndarray:
        # calculate weighted sum x1w1 + x2w2 + ... + xnwn + b
        z = x @ self.W + self.b # @ means multiply for matrix products
        return z

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.linearity(x)
        a = self.activation_function(z)
        return a
    
    def backward(self, x: np.ndarray, a: np.ndarray, y: np.ndarray, z: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        # TODO: fix matrix shapes to allow this to work
        dzdW = x.T
        dadz = self.activation_function_derivative(z)
        dLda = loss_function_derivative(a, y) # (upstream gradient)

        dLdz = dadz * dLda # delta (upstream gradient at pre activation)

        # weight gradient (chain rule)
        dLdW = dzdW @ dLdz / batch_size # .outer() to fix matrix shape mismatch

        # bias gradient (chain rule)
        dLdb = np.sum(dLdz, axis=0) / batch_size

        return dLdW, dLdb
    
    def gradient_descent(self, dLdW: np.ndarray, dLdb: np.ndarray) -> None:
        self.W -= LEARNING_RATE * dLdW
        self.b -= LEARNING_RATE * dLdb

class OutputLayer(Layer):
    def __init__(self, biases: np.ndarray, weights: np.ndarray, nodes: int) -> None:
        super().__init__(biases, weights)
    
    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return z

    def activation_function_derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)

    def linearity(self, x: np.ndarray) -> np.ndarray:
        return super().linearity(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return super().forward(x)

    def backward(self, x: np.ndarray, a: np.ndarray, y: np.ndarray, z: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        return super().backward(x, a, y, z, batch_size)

    def gradient_descent(self, dLdW: np.ndarray, dLdb: np.ndarray) -> None:
        super().gradient_descent(dLdW, dLdb)

# 10 since theres no hidden layers for now (input -> output)
# INITIALISATION
weights = np.random.normal(size=(784, 10), loc=0, scale=0.01) # mean = 0, standard deviation = 0.01 (small)
biases = np.zeros(shape=(10,))

output_layer = OutputLayer(biases, weights, 10)

# DATA
train_X, train_y, test_X, test_y = get_data()

# preprocessing for inputs: making them between 0-1 using 255 (MNIST uses grayscale)
train_X = train_X.astype(np.float32) / 255
test_X = test_X.astype(np.float32) / 255
train_y_oh = one_hot(train_y, 10)

# reduce sample size
train_X = train_X[:1000]
train_y_oh = train_y_oh[:1000]
test_X = test_X[:200]
test_y = test_y[:200]

# actual training loop for a batch
def batch_loop(n_train: int, train_X_shuffled: np.ndarray, train_y_oh_shuffled: np.ndarray) -> float:
    epoch_loss = 0.0
    for i in range(0, n_train, BATCH_SIZE):
        batch_X = train_X_shuffled[i: i + BATCH_SIZE]
        batch_y_oh = train_y_oh_shuffled[i: i + BATCH_SIZE]

        batch_size = batch_X.shape[0]

        # forward propagation -> calculate loss -> backward propagation -> gradient descent
        # forward propagation
        a = output_layer.forward(batch_X)

        # calculate loss
        loss = loss_function(a, batch_y_oh)
        epoch_loss += loss

        # backward propagation
        dLdW, dLdb = output_layer.backward(batch_X, a, batch_y_oh, z=output_layer.linearity(batch_X), batch_size=batch_size)

        # apply gradient descent
        output_layer.gradient_descent(dLdW, dLdb)
    
    # return new epoch loss
    return epoch_loss

def epoch_loop(test_X: np.ndarray, test_y: np.ndarray):
    n_train = train_X.shape[0]

    for epoch in range(1, EPOCHS):
        perm = np.random.permutation(n_train)
        train_X_shuffled = train_X[perm]
        train_y_oh_shuffled = train_y_oh[perm]

        epoch_loss = batch_loop(n_train, train_X_shuffled, train_y_oh_shuffled)
        epoch_loss /= n_train

        # evaluate on test set
        test_a = output_layer.forward(test_X)
        preds = np.argmax(test_a, axis=1)
        test_acc = np.mean(preds == test_y)

        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {epoch_loss:.6f}  test_acc: {test_acc:.4f}")

# start training NN
print("STARTING")
epoch_loop(test_X, test_y)

# 28x28 res
# INPUT: 784
# OUTPUT: 10
# start with 0 hidden layer
# then 1 hidden layer (128 nodes)

# PARAMETERS
# weights: random (very small)
# biases: zero
# how weight will be designed:
# consider layers A -> B
# B owns the weight matrix not A

# HYPER PARAMETERS
# no. hidden layers
# no. nodes per layer
# learning rate
# batch size
# loss function (how wrong the model was at a data point)
# (a - y)^2
# cost function (how wrong the model was overall)
# 1/N * sum(a - y)^2
# activation function (ReLU)

# TODO: refactor codebase and actually use NN class now (add hidden layers too)