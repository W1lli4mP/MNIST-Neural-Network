import numpy as np
from neural_network import NeuralNetwork
import json
from paths import here

def save_model(neural_network: NeuralNetwork, filename: str) -> None:
    """
    saves a model after training to models/ as a .npz (concise and efficient file format for saving models)
    """
    filepath = here("models", filename)
    weights_biases = {}
    for i, layer in enumerate(neural_network.layers):
        weights_biases[f"layer_{i}_W"] = layer.W
        weights_biases[f"layer_{i}_b"] = layer.b
    np.savez(filepath, **weights_biases)
    print(f"Model saved to {filepath}")

def load_model(neural_network: NeuralNetwork, filename: str) -> None:
    """
    loads a saved model as a .npz from models/ using a specified name
    """
    filepath = here("models", filename)
    data = np.load(filepath)
    for i, layer in enumerate(neural_network.layers):
        layer.W = data[f"layer_{i}_W"]
        layer.b = data[f"layer_{i}_b"]
    print("Model successfully loaded")

def save_results(results: dict, filename: str = "results.json") -> None:
    """
    saves a model's results as a .json to results/
    default filename is results.json
    """
    filepath = here("results", filename)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")