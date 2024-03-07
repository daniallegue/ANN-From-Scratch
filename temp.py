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
    def __init__(self, in_features, out_features, initialization, activation):
        """ Randomly initialize the weights and biases.

        Args:
            in_features: number of input features.
            out_features: number of output features.
        """

        self.weight, self.bias = initialization(in_features, out_features)
        self.cache = None
        # For storing the gradients w.r.t. the weight and the bias
        self.weight_grad = None
        self.activation = activation
        self.bias_grad = None

    def forward(self, x):
        """ Perform the forward pass of a linear layer.
        Store (cache) the input, so it can be used in the backward pass.

        Args:
            x: input of a linear layer.

        Returns:
            y: output of a linear layer.
        """
        self.cache = x
        z = x @ self.weight + self.bias
        return self.activation.forward(z)

    def backward(self, dupstream):
        """ Perform the backward pass of a linear layer.

        Args:
            dupstream: upstream gradient.

        Returns:
            dx: downstream gradient.
        """

        dupstream = self.activation.backward(dupstream)

        self.weight_grad = self.cache.T @ dupstream
        self.bias_grad = np.sum(dupstream, axis=0, keepdims=True)
        return dupstream @ self.weight.T
# %%

def uniform_initialization(f_in, f_out):
    weight = np.random.rand(f_in, f_out)
    bias = np.random.rand(1, f_out)
    return weight, bias

def normal_initialization(f_in, f_out):
    weight = np.random.randn(f_in, f_out)
    bias = np.random.randn(1, f_out)
    return weight,bias

def exp_initialization(f_in, f_out):
    weight = np.random.exponential((f_in, f_out), 2)
    bias = np.random.exponential((1, f_out), 2)
    return weight,bias

def he_initialization(f_in, f_out):
    weight = np.random.randn(f_in, f_out) * np.sqrt(2 / f_in)
    bias = np.zeros((1, f_out))
    return weight, bias

def constant_initialization(f_in, f_out):
    constant = np.random.randint(1, 10)
    weight = np.full((f_in, f_out), constant)
    bias = np.full((1, f_out), constant)
    return weight, bias


class ReLU:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        self.cache = x
        return np.maximum(0, x)

    def backward(self, dupstream):
        return dupstream * (self.cache > 0)


# %%
class SoftMax:
    def __init__(self):
        self.cache = None

    def forward(self, x):
        """ Perform a forward pass of your activation function.
        Store (cache) the output, so it can be used in the backward pass.

        Args:
            x: input to the activation function.

        Returns:
            y: output of the activation function.
        """
        shiftx = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(shiftx)
        softmax = exps / np.sum(exps, axis=1, keepdims=True)
        self.cache = softmax
        return softmax

    def backward(self, dupstream):
        """ Perform a backward pass of the activation function.
        Make sure you do not modify the original dupstream.

        Args:
            dupstream: upstream gradient.

        Returns:
            dx: downstream gradient.
        """
        softmax = self.cache
        return dupstream * softmax * (1 - softmax)

# %%
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """ Perform a forward pass over the entire network.

        Args:
            x: input data.

        Returns:
            y: predictions.
        """

        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y

    def backward(self, dupstream):
        """ Perform a backward pass over the entire network.

        Args:
            dupstream: upstream gradient.

        Returns:
            dx: downstream gradient.
        """

        dx = dupstream
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        return dx

    def optimizer_step(self, lr):
        """ Update the weight and bias parameters of each layer.

        Args:
            lr: learning rate.
        """

        for layer in self.layers:
            layer.weight = layer.weight - lr * layer.weight_grad
            layer.bias = layer.bias - lr * layer.bias_grad

# %%
def loss(y_true, y_pred):
    """ Computes the value of the loss function and its gradient.

    Args:
        y_true: ground truth labels.
        y_pred: predicted labels.

    Returns:
        loss: value of the loss.
        grad: gradient of loss with respect to the predictions.
    """

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    grad = y_pred - y_true
    return loss, grad

# %%
def one_hot_encode(y, num_classes):
    """Converts a vector of labels to one-hot encoded format.

    Args:
        y: Array of labels, shape (num_samples,).
        num_classes: Number of classes.

    Returns:
        One-hot encoded labels, shape (num_samples, num_classes).
    """
    y = y.astype(int) - 1

    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot

# %%
def train(net, inputs, labels, criterion, lr, epochs):
    losses = []
    accuracies = []

    for _ in range(epochs):
        predictions = net.forward(inputs)
        loss, grad = criterion(labels, predictions)
        net.backward(grad)
        net.optimizer_step(lr)

        losses.append(loss)

        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(labels, axis=1)  # Adjust if `labels` is already not one-hot
        accuracy = np.sum(pred_labels == true_labels) / len(labels)
        accuracies.append(accuracy)

    return losses, accuracies

# %%
combined_data = np.column_stack((X_data, y_data))
np.random.shuffle(combined_data)
features = combined_data[:, :-1]
targets = combined_data[:, -1]

split_index = int(0.7 * features.shape[0])
X_train = features[:split_index]
# X_train = X_train.transpose()
X_test = features[split_index:]
X_test = X_test.transpose() # TODO: transpose?
y_train = targets[:split_index]
y_train = one_hot_encode(y_train, 7)
# y_train = y_train.transpose()
y_test = targets[split_index:]
y_test = y_test.transpose()

# %%
epochs = 25

initialization_functions = [(normal_initialization, 'Normal Distribution'),
                            (exp_initialization, 'Exponential Distribution'),
                            (uniform_initialization, 'Uniform Distribution'), (constant_initialization, 'Constant'),
                            (he_initialization, 'He')]

relu = ReLU()
softmax = SoftMax()

layers = [Layer(10, 8, he_initialization,relu ), Layer(8, 8, he_initialization, relu), Layer(8, 8, he_initialization, relu), Layer(8, 7, normal_initialization, softmax)]
network = Network(layers)
losses, accuracies = train(network, X_train, y_train, loss, 0.0001, epochs)



