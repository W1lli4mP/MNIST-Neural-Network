from paths import here
from mnist import MNIST
import numpy as np

LEARNING_RATE = 0.01
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
    def __init__(self, layer_dims: list[int]):
        # layer dims is a list compiled of the number of nodes in each layer (e.g. 784, 128, 10 means 784 inputs, 128 hidden, 10 outputs)
        self.layers = []
        for i in range(len(layer_dims) - 1):
            in_features = layer_dims[i]
            out_features = layer_dims[i + 1]

            weights = np.random.normal(size=(in_features, out_features), loc=0, scale=0.01) # mean = 0, standard deviation = 0.01 (small)
            biases = np.zeros(out_features)

            # if output layer, initialise it
            if i == len(layer_dims) - 2:
                self.layers.append(OutputLayer(biases, weights, out_features))
            else:
                self.layers.append(Layer(biases, weights))
    
    def forward(self, x):
        self.activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            self.activations.append(x)
        return x
    
    def backward(self, x, y, batch_size):
        dLda = loss_function_derivative(self.activations[-1], y) # obtain last activation (from output layer)

        # traverse layers in reverse (its backprop)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layer_input = self.activations[i] # activation from previous layer
            
            # compute gradients and apply gradient descent
            z = layer.linearity(layer_input)
            dLda = layer.backward(layer_input, z, dLda, batch_size) # calculate new upstream gradient and apply it via backpropagation

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
    
    def backward(self, x: np.ndarray, z: np.ndarray, dLda: np.ndarray, batch_size: int) -> np.ndarray:
        """
        dLda is the upstream gradient
        returns dLda_prev which is the upstream gradient of the previous layer
        """
        # TODO: fix matrix shapes to allow this to work
        dadz = self.activation_function_derivative(z)
        # dLda = loss_function_derivative(a, y) # (upstream gradient)

        dLdz = dadz * dLda # delta (upstream gradient at pre activation)

        # weight gradient (chain rule)
        dzdW = x.T
        dLdW = dzdW @ dLdz / batch_size # .outer() to fix matrix shape mismatch

        # bias gradient (chain rule)
        dLdb = np.sum(dLdz, axis=0) / batch_size

        dLda_prev = dLdz @ self.W.T

        # apply gradient descent
        self.gradient_descent(dLdW, dLdb)
        return dLda_prev
    
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

    def backward(self, x: np.ndarray, z: np.ndarray, dLda: np.ndarray, batch_size: int) -> np.ndarray:
        return super().backward(x, z, dLda, batch_size)

    def gradient_descent(self, dLdW: np.ndarray, dLdb: np.ndarray) -> None:
        super().gradient_descent(dLdW, dLdb)

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

def main():
    # INITIALISE NN
    nn = NeuralNetwork([784, 128, 10])

    # EXTRACTING THE DATA
    train_X, train_y, test_X, test_y = get_data()

    # preprocessing for inputs: making them between 0-1 using 255 (MNIST uses grayscale)
    train_X = train_X.astype(np.float32) / 255
    test_X = test_X.astype(np.float32) / 255
    train_y_oh = one_hot(train_y, 10)

    # reduce sample size
    # train_X = train_X[:1000]
    # train_y_oh = train_y_oh[:1000]
    # test_X = test_X[:200]
    # test_y = test_y[:200]

    # start training NN
    print("STARTING")
    epoch_loop(nn, train_X, train_y_oh, test_X, test_y)

main()