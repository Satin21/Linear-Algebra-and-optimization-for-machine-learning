# Neural networks
import numpy as np
from Layer import Layer


class NN:
    def __init__(self, n_in, n_out, n_layers, n_per_layer, batch_size=16):
        # self.batch_size = batch_size

        # Define the NN
        self.layers = []
        self.layers.append(Layer(n_per_layer[0], n_in))  # Define the first hidden layer
        for i in range(1, n_layers):
            self.layers.append(Layer(n_per_layer[i], n_per_layer[i - 1]))  # Define the other hidden layers
        self.layers.append(Layer(n_out, n_per_layer[n_layers - 1], None))  # The output layer

    # Propagate the input x through the network
    def __call__(self, x):
        print(x.shape)
        if len(x.shape) == 1:
            x = np.reshape(x, (len(x), 1))
        for l in self.layers:
            x = l(x)

        return x

    # Learn by doing back propagation
    def learn(self):

