import numpy as np
import time

def initialize_matrix(shape):
    return np.random.rand(*shape).astype(np.float32)

def relu(x):
    return np.maximum(0, x)

def mlp(input, input_layer_weights, weights, biases, output_layer_weights):
    # Initial forward pass through the input layer
    x = input @ input_layer_weights
    x = relu(x)  # ReLU activation after input layer

    # Forward pass through hidden layers
    for i in range(len(weights)):
        x = x @ weights[i]  # Matrix multiplication for each layer
        x += biases[i]  # Add bias for the layer
        x = relu(x)  # ReLU activation for the layer

    # Final forward pass through the output layer
    output = x @ output_layer_weights
    return output

# Define layer sizes
B = 1  # Batch size
input_dim = 768
output_dim = 32000
hidden_size = 2048
num_layers = 32

# Initialize input and weights
input = initialize_matrix((B, input_dim))
input_layer_weights = initialize_matrix((input_dim, hidden_size))
weights = [initialize_matrix((hidden_size, hidden_size)) for _ in range(num_layers)]
biases = [initialize_matrix((1, hidden_size)) for _ in range(num_layers)]
output_layer_weights = initialize_matrix((hidden_size, output_dim))

# Measure execution time
start_time = time.time()
output = mlp(input, input_layer_weights, weights, biases, output_layer_weights)
end_time = time.time()

print(f"MLP execution time: {end_time - start_time:.6f} seconds")
