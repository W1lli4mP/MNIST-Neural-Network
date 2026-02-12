import numpy as np
from config import LEARNING_RATE

# activation and loss functions/derivatives
def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def loss_function(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum((a - y) ** 2)

def loss_function_derivative(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (a - y) # with respect to a

# new activation and loss functions
def softmax(z: np.ndarray) ->np.ndarray:
    # softmax depends on relative differences, so reduce the number size by subtracting by the maximum to avoid stuff like (inf/ inf)
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z) # e^z
    exp_sum = np.sum(ez, axis=-1, keepdims=True) # axis = -1 sums up all of the classes to 1 and not anything else
    return ez / exp_sum

def cross_entropy(a: np.ndarray, y: np.ndarray) -> np.ndarray:
    epsilon = 1e-10 # to avoid log(0)
    return -np.sum(y * np.log(a + epsilon))

# neural network and layer classes
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
        return softmax(z)

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