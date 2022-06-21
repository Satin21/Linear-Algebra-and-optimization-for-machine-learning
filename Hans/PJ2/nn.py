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
    def __init__(self, n_in: int, n_out: int, n_layers: int, n_per_layer: list, lr=.01, batch_size=16):  # TODO: hidden_act_fu, out_act_fu
        # self.batch_size = batch_size

        # Define the NN
        self.layers = []
        self.layers.append(Layer(n_per_layer[0], n_in, 'sigmoid', lr=lr))  # Define the first hidden layer
        for i in range(1, n_layers):
            self.layers.append(
                Layer(n_per_layer[i], n_per_layer[i - 1], 'sigmoid', lr=lr))  # Define the other hidden layers
        self.layers.append(Layer(n_out, n_per_layer[n_layers - 1], None, lr=lr))  # TODO: activation = None? The output layer

    # Propagate the input x through the network
    def __call__(self, x: np.ndarray):
        # Ensure that x is a vector to process it in the layers
        if len(x.shape) == 1:
            x = np.reshape(x, (len(x), 1))
        for layer in self.layers:
            x = layer(x)
        self.y_hat = x

        return x

    # Learn by doing back-propagation
    def learn(self, y_true):

        # Compute the gradients
        err_grad = mse_grad(y_true, self.y_hat)  # TODO: change the loss function and define the corresponding gradient
        for layer in self.layers:
            print('err_grad:', err_grad)
            err_grad = layer.compute_grad(err_grad)  # TODO: use err_grad correctly for back-propagation through the layers

        # Update all weights
        for l in self.layers:
            l.update_weights()
