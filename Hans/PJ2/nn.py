# Neural networks
import numpy as np
from Layer import Layer


def mse_grad(y_true, y_pred):  # MSE derivative
    if isinstance(y_true, float):
        n = 1
    else:
        n = y_true.shape[0]

    return 2 * (y_pred - y_true) / n


class NN:
    def __init__(self, n_in: int, n_out: int, n_layers: int, n_per_layer: list, lr=.01, batch_size=16):
        # self.batch_size = batch_size

        # Define the NN
        self.layers = []
        self.layers.append(Layer(n_per_layer[0], n_in, lr=lr))  # Define the first hidden layer
        for i in range(1, n_layers):
            self.layers.append(
                Layer(n_per_layer[i], n_per_layer[i - 1], 'sigmoid', lr=lr))  # Define the other hidden layers
        self.layers.append(Layer(n_out, n_per_layer[n_layers - 1], None, lr=lr))  # The output layer

    # Propagate the input x through the network
    def __call__(self, x: list):
        # print(x.shape)
        if len(x.shape) == 1:
            x = np.reshape(x, (len(x), 1))
        for l in self.layers:
            x = l(x)
        self.y_hat = x

        return x

    # Learn by doing back propagation
    def learn(self, y_true):
        # Compute the gradients
        err_grad = mse_grad(y_true, self.y_hat)
        for l in self.layers:
            print('err_grad:', err_grad)
            err_grad = l.compute_grad(err_grad)  # TODO: use err_grad correctly for back-propagation through the layers

        # Update all weights
        for l in self.layers:
            l.update_weights()
