import numpy as np
from config import LEARNING_RATE

# activation functions
def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def softmax(z: np.ndarray) ->np.ndarray:
    # softmax depends on relative differences, so reduce the number size by subtracting by the maximum to avoid stuff like (inf/ inf)
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z) # e^z
    exp_sum = np.sum(ez, axis=-1, keepdims=True) # axis = -1 sums up all of the classes to 1 and not anything else
    return ez / exp_sum

# softmax derivative is not required due to delta being simplified with cross-entropy derivative

# loss functions
def mean_squared_error(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((y_hat - y) ** 2)

def mean_squared_error_derivative(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2 * (y_hat - y)

def cross_entropy(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    epsilon = 1e-10 # to avoid log(0)
    return -np.sum(y * np.log(y_hat + epsilon))

# cross-entropy derivative is not required due to delta being simplified with softmax derivative

# activation and loss configuration
def hidden_activation_function(z: np.ndarray) -> np.ndarray:
    return ReLU(z)

def hidden_activation_function_derivative(z: np.ndarray) -> np.ndarray:
    return ReLU_derivative(z)

def output_activation_function(z: np.ndarray) -> np.ndarray:
    return softmax(z)

def output_activation_function_derivative(z: np.ndarray) -> np.ndarray:
    return # don't need it for softmax + cross-entropy

def loss_function(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((a - y) ** 2)

def loss_function_derivative(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (a - y) # with respect to a

# neural network and layer classes
class NeuralNetwork:
    def __init__(self, layer_dims: list[int]) -> None:
        # layer dims is a list compiled of the number of nodes in each layer (e.g. 784, 128, 10 means 784 inputs, 128 hidden, 10 outputs)
        self.layers = []
        for i in range(len(layer_dims) - 1):
            in_features = layer_dims[i]
            out_features = layer_dims[i + 1]

            weights = np.random.normal(size=(in_features, out_features), loc=0, scale=0.01) # mean = 0, standard deviation = 0.01 (small)
            biases = np.zeros(out_features)

            # if output layer, initialise it
            if i == len(layer_dims) - 2:
                self.layers.append(OutputLayer(biases, weights))
            else:
                self.layers.append(Layer(biases, weights))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.activations = [x]
        for layer in self.layers:
            x = layer.forward(x)
            self.activations.append(x)
        return x
    
    def backward(self, y: np.ndarray, batch_size: int) -> None:
        # do backpropagation + gradient descent on output layer first
        output_layer = self.layers[-1]

        ## retrieve the output layer's activation (called y_hat instead of a)
        y_hat = self.activations[-1]

        # get previous layer's activation
        a_prev = self.activations[-2]

        # get upstream gradient
        dLda = output_layer.backward(y_hat, a_prev, y, batch_size)

        # traverse HIDDEN layers in reverse order for backpropagation
        # NOTE: directions are relative to the forward direction despite traversal order
        for i in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[i]

            # retrieve activation from the previous layer's activations
            a_prev = self.activations[i]
            a = self.activations[i + 1]
            
            # backpropagate to find upstream gradient (pre-gradient descent) for preceding layer and apply gradient descent
            dLda = current_layer.backward(a, a_prev, dLda, batch_size)

class Layer:
    def __init__(self, biases: np.ndarray, weights: np.ndarray) -> None:
        self.b = biases
        self.W = weights

    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return hidden_activation_function(z)

    def activation_function_derivative(self, z: np.ndarray) -> np.ndarray:
        return ReLU_derivative(z)

    def get_preactivation(self, x: np.ndarray) -> np.ndarray:
        # calculate weighted sum x1w1 + x2w2 + ... + xnwn + b
        z = x @ self.W + self.b # @ means multiply for matrix products
        return z

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = self.get_preactivation(x)
        a = self.activation_function(z)
        return a
    
    def backward(self, a: np.ndarray, x: np.ndarray, dLda: np.ndarray, batch_size: int) -> np.ndarray:
        ## compute delta
        # dLdz = dLda * dadz
        z = self.get_preactivation(x)
        delta = dLda * self.activation_function_derivative(z)

        ## calculate upstream gradient for previous layer (pre-gradient descent)
        # dLda_prev = dLda * dadz * W^T
        dLda_prev = delta @ self.W.T

        ## calculate components for gradient descent
        # per-sample bias gradient (dLdb)
        # dLdb = delta * dzdb
        # dLdb = delta * 1
        dLdb = delta

        # per-sample weight gradient (dLdW)
        # dLdW = dLdz * dzdW
        # dLdW = delta * x
        dLdW = x.T @ delta # other way round due to shapes

        ## calculate mini-batch gradient
        # bias component
        dLdb_batch = np.sum(dLdb, axis=0) / batch_size

        # weight component
        dLdW_batch = dLdW / batch_size

        ## apply mini-batch gradient descent
        self.gradient_descent(dLdW_batch, dLdb_batch)

        ## backpropagate upstream gradient to preceding layer
        return dLda_prev
    
    def gradient_descent(self, dLdW: np.ndarray, dLdb: np.ndarray) -> None:
        self.W -= LEARNING_RATE * dLdW
        self.b -= LEARNING_RATE * dLdb

class OutputLayer(Layer):
    def __init__(self, biases: np.ndarray, weights: np.ndarray) -> None:
        super().__init__(biases, weights)
    
    def activation_function(self, z: np.ndarray) -> np.ndarray:
        return output_activation_function(z)

    def activation_function_derivative(self, z: np.ndarray) -> np.ndarray:
        return output_activation_function_derivative(z)

    def get_preactivation(self, x: np.ndarray) -> np.ndarray:
        return super().get_preactivation(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return super().forward(x)

    def backward(self, a: np.ndarray, x: np.ndarray, y: np.ndarray, batch_size: int) -> np.ndarray:
        ## compute delta (dLdz) (NOTE: HARDCODED FOR SOFTMAX + CROSS-ENTROPY)
        # dLdz = dLda * dadz
        delta = a - y

        ## calculate upstream gradient for previous layer (pre-gradient descent)
        # dLda_prev = dLda * dadz * W^T
        # dLda_prev = delta * W^T
        dLda_prev = delta @ self.W.T

        ## calculate components for gradient descent
        # per-sample bias gradient (dLdb)
        # dLdb = dLdz * dzdb
        # dLdb = delta * 1
        dLdb = delta

        # per-sample weight gradient (dLdW)
        # dLdW = dLdz * dzdW
        # dLdW = delta * x
        dLdW = x.T @ delta # other way round due to shapes

        ## calculate mini-batch gradient
        # bias component
        dLdb_batch = np.sum(dLdb, axis=0) / batch_size

        # weight component
        dLdW_batch = dLdW / batch_size

        ## apply mini-batch gradient descent
        self.gradient_descent(dLdW_batch, dLdb_batch)

        ## backpropagate upstream gradient to preceding layer
        return dLda_prev

    def gradient_descent(self, dLdW: np.ndarray, dLdb: np.ndarray) -> None:
        super().gradient_descent(dLdW, dLdb)