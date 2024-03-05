# %%
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Read data
X_data = pd.read_csv('data/features.txt', header=None).to_numpy()
y_data = pd.read_csv('data/targets.txt', header=None).to_numpy()
y_unknown = pd.read_csv('data/unknown.txt', header=None).to_numpy()

# %%
# Q1 - Architecture
class SinglePerceptron:
    def __init__(self, size):
        # Intialize random weights and bias
        self.weights = np.random.rand(size)
        self.bias = np.random.rand(1)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.bias

    def activation(self, x):
        # ReLU
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(self.forward(inputs))

    def train(self, X, y, epochs, lr):
        error_per_epoch = []
        for _ in range(epochs):
            for i in range(X.shape[0]):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                error_per_epoch.append(error)
                self.weights += lr * error * X[i]
                self.bias += lr * error
        return error_per_epoch

# Train the perceptron of the logic gates
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])

perceptron = SinglePerceptron(2)
errors_and = perceptron.train(X, y_and, 15, 1)
perceptron = SinglePerceptron(2)
errors_or = perceptron.train(X, y_or, 15, 1)
perceptron = SinglePerceptron(2)
errors_xor = perceptron.train(X, y_xor, 15, 1)

plt.plot(errors_and, label='AND')
plt.show()
plt.plot(errors_or, label='OR')
plt.show()
plt.plot(errors_xor, label='XOR')
plt.show()

# %%
# Q2 - How many input neurons we need?
# We need 10 input neurons, each representing 1 feature of the input data

# %%
# Q3 - How many output neurons we need?
# Since there are 7 possible classes, we need 7 output neurons

# %%
# Q4 - How many hidden layers we need?
# My initial guess is that we need to hidden layers. (Expand on this answer)

# %%
# Q5 - What activation function would you use for the output layer?
# Since this is multi-class classification problem, the output will be a 7-vector of probabilities. Hence, we need to output the class with the largest probability. Therefore, we should use the softmax activation function.

# %%
# Q6 - Create an schematic diagram of the architecture
# DONE

# %%
# Q7 - Train the model
# Since we want unbiased estimates and good performing predictions, we will make a 70/30 split
split_index = int(0.7 * X_data.shape[0])
X_train = X_data[:split_index]
X_train = X_train.transpose()
X_test = X_data[split_index:]
X_test = X_test.transpose()
y_train = y_data[:split_index]
y_train = y_train.transpose()
y_test = y_data[split_index:]
y_test = y_test.transpose()

# %%
# Q8 - How to evaluate the performance of the network?
# We will use the accuracy metric to evaluate the performance of the network. This is because we are dealing with a classification problem and we want to know how many of the predictions were correct.

# %%
# Q9 - When do we decide to stop training?
# We will stop training after a certain number of iterations (epochs), or after the error stops decreasing or the error is below a certain threshold.

# %%
# Define Layer class
class Layer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = activation

    def activate(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'softmax':
            e_x = np.exp(x - np.max(x, axis=0))
            return e_x / np.sum(e_x, axis=0)
        return x

    def derivative(self, x):
        if self.activation == 'sigmoid':
            return x * (1 - x)
        elif self.activation == 'relu':
            return np.where(x <= 0, 0, 1)
        elif self.activation == 'softmax':
            return x * (1 - x)  # Note: This is not used directly, handled in cross-entropy derivative
        return 1


class MultilayerPerceptron:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, X):
        activation = X
        activations = [X]  # List of all activations, layer by layer
        zs = []  # List of all z vectors, layer by layer

        for layer in self.layers:
            z = np.dot(layer.weights, activation) + layer.biases
            activation = layer.activate(z)
            zs.append(z)
            activations.append(activation)

        return activations, zs

    def backpropagation(self, X, y):
        nabla_b = [np.zeros(layer.biases.shape) for layer in self.layers]
        nabla_w = [np.zeros(layer.weights.shape) for layer in self.layers]

        activations, zs = self.forward_pass(X)
        delta = self.cost_derivative(activations[-1], y) * self.layers[-1].derivative(activations[-1])
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True)  # Sum over samples for biases
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, len(self.layers) + 1):
            z = zs[-l]
            sp = self.layers[-l].derivative(activations[-l])
            delta = np.dot(self.layers[-l + 1].weights.transpose(), delta) * sp
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True)  # Similarly sum over samples
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def update_parameters(self, X, y, eta):
        nabla_b, nabla_w = self.backpropagation(X, y)
        for l in range(len(self.layers)):
            self.layers[l].weights -= eta * nabla_w[l]
            self.layers[l].biases -= eta * nabla_b[l]

    def train(self, X, y, epochs, eta):
        for epoch in range(epochs):
            # Update model using the entire dataset
            self.update_parameters(X, y, eta)
            print(f"Epoch {epoch} complete")

    def update_parameters(self, X, y, eta):
        # Compute gradients for the whole dataset
        nabla_b, nabla_w = self.backpropagation(X, y)
        for l in range(len(self.layers)):
            self.layers[l].weights -= eta * nabla_w[l]
            self.layers[l].biases -= eta * nabla_b[l]

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

# %%
# Create instance of MLP
# 2 Hidden Layers with 5 neurons each
mlp = MultilayerPerceptron([Layer(10, 5, 'relu'), Layer(5, 5, 'relu'), Layer(5, 7, 'softmax')])
losses = mlp.train(X_train, y_train, 100, 0.1)






