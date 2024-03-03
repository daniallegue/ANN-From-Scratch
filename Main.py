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
#errors_and = perceptron.train(X, y_and, 15, 1)
#errors_or = perceptron.train(X, y_or, 15, 1)
errors_xor = perceptron.train(X, y_xor, 15, 1)

# plt.plot(errors_and, label='AND')
# plt.show()
# plt.plot(errors_or, label='OR')
# plt.show()
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
# TODO (Diagram to be created)

# %%
# Q7 - Train the model
# Since we want unbiased estimates and good performing predictions, we will make a 70/30 split. TODO (To add more)
split_index = int(0.7 * X_data.shape[0])
X_train = X_data[:split_index]
# X_train = X_train.transpose()
X_test = X_data[split_index:]
X_test = X_test.transpose()
y_train = y_data[:split_index]
# y_train = y_train.transpose()
y_test = y_data[split_index:]
y_test = y_test.transpose()

# %%
# Q8 - How to evaluate the performance of the network?
# We will use the accuracy metric to evaluate the performance of the network. This is because we are dealing with a classification problem and we want to know how many of the predictions were correct.

# %%
# Q9 - When do we decide to stop training?
# We will stop training after a certain number of iterations (epochs), or after the error stops decreasing or the error is below a certain threshold.

# %%
# Define multi-layer perceptron
class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self._init_params()

    def _init_params(self):
        for i in range(len(self.layer_sizes) - 1):
            weight = np.random.randn(self.layer_sizes[i], self.layer_sizes[i + 1]) * np.sqrt(2. / self.layer_sizes[i])
            bias = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def _relu(self, Z):
        return np.maximum(0, Z)

    def _softmax(self, Z):
        e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return e_Z / np.sum(e_Z, axis=1, keepdims=True)

    def _relu_derivative(self, Z):
        return Z > 0

    def _forward(self, X):
        A = X
        cache = []
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._relu(Z)
            cache.append((A, Z))
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        A = self._softmax(Z)
        cache.append((A, Z))
        return A, cache

    def _backward(self, X, Y, cache):
        m = Y.shape[0]
        gradients = []
        A_final, Z_final = cache[-1]
        dZ = A_final - Y  # For softmax with cross-entropy loss
        for i in reversed(range(len(self.weights))):
            A_prev, Z = cache[i - 1] if i > 0 else (X, None)
            dW = np.dot(A_prev.T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self._relu_derivative(Z)
            gradients.insert(0, (dW, db))
        return gradients

    def _update_params(self, gradients, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * gradients[i][0]
            self.biases[i] -= lr * gradients[i][1]

    def train(self, X, Y, epochs, lr):
        losses = []
        for epoch in range(epochs):
            A, cache = self._forward(X)
            gradients = self._backward(X, Y, cache)
            self._update_params(gradients, lr)
            loss = -np.mean(Y * np.log(A + 1e-7))  # Cross-entropy loss
            losses.append(loss)

        return losses


    def predict_proba(self, X):
        A, _ = self._forward(X)
        return A

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

# %%
# Create instance of MLP
# 2 Hidden Layers with 5 neurons each
mlp = MLP([10, 5, 7])
losses = mlp.train(X_train, y_train, 100, 0.1)






