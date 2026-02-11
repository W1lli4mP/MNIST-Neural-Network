# metadata
MODEL_NAME = "prototype_4.npz"

# hyperparameters for the neural network
LAYER_DIMS = [784, 128, 10]
INPUT_UNITS = LAYER_DIMS[0]
HIDDEN_UNITS = sum(LAYER_DIMS[1:-1])
OUTPUT_UNITS = LAYER_DIMS[-1]

LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 100