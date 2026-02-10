import numpy as np
from neural_network import NeuralNetwork

def save_model(neural_network: NeuralNetwork, filename: str) -> None:
    filepath = f"../models/{filename}"
    weights_biases = {}
    for i, layer in enumerate(neural_network.layers):
        weights_biases[f"layer_{i}_W"] = layer.W
        weights_biases[f"layer_{i}_b"] = layer.b
    np.savez(filepath, **weights_biases)
    print(f"Model saved to {filepath}")

def load_model(neural_network: NeuralNetwork, filename: str) -> None:
    filepath = f"../models/{filename}"
    data = np.load(filepath)
    for i, layer in enumerate(neural_network.layers):
        layer.W = data[f"layer_{i}_W"]
        layer.b = data[f"layer_{i}_b"]
    print("SUCCESSFULLY LOADED THE MODEL")
