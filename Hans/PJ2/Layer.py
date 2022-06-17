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

    return 2 * (y_pred - y_true) / N


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
    def __init__(self, n, n_prev, activation='relu', lr=.01, beta1=.9, beta2=.99):
        self.lr = lr  # Set the learning rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = 1.  # = pow(beta1, t) in the t-th iteration/call of compute_grad()
        self.beta2_t = 1.

        # Set the activation function
        act_map = {'relu': relu, 'sigmoid': sigmoid, None: identity}
        if activation not in act_map:
            ValueError('Activation function ' + self.activation + ' is unknown.')
        self.activate = act_map[activation]

        # Set the derivative thereof
        act_dash_map = {'relu': relu_dash, 'sigmoid': sigmoid_dash, None: identity}
        self.activate_dash = act_dash_map[activation]

        # Initialize W and b (normalized initialization on W)
        u = np.sqrt(6 / (n + n_prev))
        self.W = np.random.uniform(-u, u, (n, n_prev))
        self.b = np.zeros((n, 1))
        self.grad_W = None
        self.grad_b = None
        self.W_m = np.zeros((n, n_prev))
        self.W_v = np.zeros((n, n_prev))
        self.b_m = np.zeros((n, 1))
        self.b_v = np.zeros((n, 1))

    # Propagate the input data through this layer & return the result
    def __call__(self, x):
        # x = normalize(x)
        # print(self.W @ x)
        self.last_in = x
        self.last_out = self.W @ x + self.b
        y = self.activate(self.last_out)
        if len(y) == 1:
            return y[0, 0]
        return y

    # Compute the gradient of the loss function w.r.t. the parameters.
    # Inspired by https://medium.com/@neuralthreads/backpropagation-made-super-easy-for-you-part-1-6fb4aa5a0aaf
    # TODO Implement ADAM GD, see p.105
    def compute_grad(self, err_grad):
        grad_W = err_grad * self.activate_dash(self.last_out).dot(self.last_in.T)
        grad_b = err_grad * self.activate_dash(self.last_out)
        new_err_grad = np.sum(err_grad * self.activate_dash(self.last_out) * self.W, axis=0).reshape((-1, 1))

        self.W_m = self.beta1 * self.W_m + (1 - self.beta1) * grad_W
        self.W_v = self.beta2 * self.W_v + (1 - self.beta2) * np.square(grad_W)
        self.b_m = self.beta1 * self.b_m + (1 - self.beta1) * grad_b
        self.b_v = self.beta2 * self.b_v + (1 - self.beta2) * np.square(grad_b)

        return new_err_grad

    def update_weights(self):
        self.beta1_t *= self.beta1
        self.beta1_t *= self.beta2
        W_m_hat = self.W_m / (1 - self.beta1_t)
        W_v_hat = self.W_v / (1 - self.beta2_t)
        b_m_hat = self.b_m / (1 - self.beta1_t)
        b_v_hat = self.b_v / (1 - self.beta2_t)
        eps = 1E-6
        self.W += -self.lr / (np.sqrt(W_v_hat) + eps) * W_m_hat
        self.b += -self.lr / (np.sqrt(b_v_hat) + eps) * b_m_hat
