import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_dash(x):  # Sigmoid derivative
    y = sigmoid(x)
    return y * (1 - y)


def relu(x, leak=0):  # ReLU
    return np.where(x <= 0, leak * x, x)


def relu_dash(x, leak=0):  # ReLU derivative
    return np.where(x <= 0, leak, 1)


def identity(x):
    return x


def mse(y_true, y_pred):  # MSE
    return np.mean((y_true - y_pred) ** 2)


def mse_grad(y_true, y_pred):  # MSE derivative
    N = y_true.shape[0]

    return -2 * (y_true - y_pred) / N


def normalize(x):  # Layer Normalization
    mean = x.mean()
    std = x.std()

    return (x - mean) / (std + 10 ** -100)


# def cross_E(y_true, y_pred):  # CE
#     return -np.sum(y_true * np.log(y_pred + 10 ** -100))
#
#
# def cross_E_grad(y_true, y_pred):  # CE derivative
#     return -y_true / (y_pred + 10 ** -100)


# From https://medium.com/@neuralthreads/layer-normalization-applied-on-a-neural-network-f6ad51341726
def normalize_dash(x):  # Normalization derivative
    N = len(x)
    I = np.eye(N)
    mean = x.mean()
    std = x.std()

    return ((N * I - 1) / (N * std + 10 ** -100)) - (((x - mean).dot((x - mean).T)) / (N * std ** 3 + 10 ** -100))


class Layer:
    def __init__(self, n, n_prev, activation='relu'):
        self.n = n  # #Neurons in the layer

        # Set the activation function
        map = {'relu': relu, 'sigmoid': sigmoid, None: identity}
        if activation not in map:
            ValueError('Activation function ' + self.activation + ' is unknown.')
        self.activate = map[activation]

        # Initialize W and b (normalized initialization on W)
        u = np.sqrt(6 / (n + n_prev))
        self.W = np.random.uniform(-u, u, (n, n_prev))
        self.b = np.zeros((n, 1))

    # Propagate the input data through this layer & return the result
    def __call__(self, x):
        # x = normalize(x)
        print(self.W @ x)
        y = self.activate(self.W @ x + self.b)
        if len(y) == 1:
            return y[0, 0]
        return y

    def learn(self, lr):
        grad_w = mse_grad(y, y_hat) * sig_dash(in_output_layer).dot(out_hidden_2.T)
        self.W += -lr * grad_w
        grad_b = mse_grad(y, y_hat) * sig_dash(in_output_layer)
        self.b += -lr * grad_b

        error_grad_upto_H2 = np.sum(mse_grad(y, y_hat) * sig_dash(in_output_layer) * w3, axis=0).reshape((-1, 1))# error grad upto H2

        grad_w2 = error_grad_upto_H2 * sig_dash(in_hidden_2).dot(out_hidden_1.T)        # grad w2
        grad_b2 = error_grad_upto_H2 * sig_dash(in_hidden_2)  # grad b2
