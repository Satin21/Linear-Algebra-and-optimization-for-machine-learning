# Neural networks
import numpy as np
from Layer import Layer


def mse_der(y_true, y_pred):  # MSE derivative
    if isinstance(y_true, float):
        n = 1
    else:
        n = y_true.shape[0]

    return 2 * (y_pred - y_true) / n


# Compute the loss function: binary cross-entropy function
# loss = sum ( y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) )
def loss(y_true: list, y_pred: list):
    return cross_e(y_true, y_pred) + cross_e(1 - y_true, 1 - y_pred)


# Compute the gradient/derivative of the binary regression loss function
def loss_der(y_true, y_pred):
    return -sum(y_true // y_pred - [1 - y for y in y_true] // (1 - y_pred))


# Define the binary cross-entropy (CE) function
def cross_e(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 10 ** -100))


# The CE derivative
def cross_e_der(y_true, y_pred):
    return -y_true / (y_pred + 10 ** -100)


class NN:
    # Define/construct the NN
    # n_in = #inputs, n_out = #outputs, n_layers = #layers, n_per_layer = a list containing #neurons for each layer
    # hidden_activation = the activation function for all hidden layers, out_activation = the same for the output layer,
    # lr = the learning rate for the gradient updates.
    def __init__(self, n_in: int, n_out: int, n_layers: int, n_per_layer: list, hidden_activation='relu',
                 out_activation='sigmoid', lr=.01, optimizer='adam'):
        self.layers = []
        self.layers.append(
            Layer(n_per_layer[0], n_in, hidden_activation, lr=lr, optimizer=optimizer))  # Define the first hidden layer
        for i in range(1, n_layers):
            self.layers.append(Layer(n_per_layer[i], n_per_layer[i - 1], hidden_activation, lr=lr,
                                     optimizer=optimizer))  # Define the other hidden layers
        self.layers.append(Layer(n_out, n_per_layer[n_layers - 1], out_activation, lr=lr,
                                 optimizer=optimizer))  # Define the output layer

    # Forward-Propagation: Propagate the input x through the network and return the output
    def __call__(self, x: np.ndarray):

        # Ensure that x is a vector to process it in the layers
        if len(x.shape) == 1:
            x = np.reshape(x, (len(x), 1))

        # Do the forward propagations
        for layer in self.layers:
            x = layer(x)

        # The output
        self.y_pred = x

        return x

    # Do the backward-propagation
    def learn(self, y_true: list):

        # Compute the gradients
        # err_grad = loss_der(y_true, self.y_pred)
        err_grad = mse_der(y_true, self.y_pred)
        for layer in reversed(self.layers):
            err_grad = layer.compute_grad(err_grad)

        # Update all weights
        for layer in self.layers:
            layer.update_weights()
