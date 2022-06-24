import numpy as np


# The sigmoid actiovation function
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# The sigmoid derivative
def _sigmoid_der(x):
    y = _sigmoid(x)
    return y * (1 - y)


# The ReLU activation function
def _relu(x, leak=0):
    return np.where(x <= 0, leak * x, x)


# The ReLU derivative
def _relu_der(x, leak=0):
    return np.where(x <= 0, leak, 1)


def _identity(x):
    return x


# Compute the MSE
def _mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# Compute the MSE derivative
def _mse_der(y_true, y_pred):
    N = y_true.shape[0]

    return 2 * (y_pred - y_true) / N


# NICE TODO: use batch normalization per layer
def _normalize(x):  # Layer Normalization
    mean = x.mean()
    std = x.std()

    return (x - mean) / (std + 10 ** -100)


# From https://medium.com/@neuralthreads/layer-normalization-applied-on-a-neural-network-f6ad51341726
def _normalize_der(x):  # Normalization derivative
    N = len(x)
    I = np.eye(N)
    mean = x.mean()
    std = x.std()

    return ((N * I - 1) / (N * std + 10 ** -100)) - (((x - mean).dot((x - mean).T)) / (N * std ** 3 + 10 ** -100))


class Layer:
    def __init_weights(self, n, n_prev, activation):

        # Initialize W and b (normalized initialization on W)
        if activation == 'sigmoid':
            u = np.sqrt(6. / (n + n_prev))
        elif activation == 'relu':
            u = 1. / np.sqrt(n)
        self.W = np.random.uniform(-u, u, (n, n_prev))
        self.b = np.zeros((n, 1))

    # Define a layer comprising n neurons, n_prev = #neurons in the previous layer, learning rate lr,
    # optimizer 'adam' (Adam gradient) or 'gd' (standard GD), beta1 and beta2 are Adam parameters
    def __init__(self, n, n_prev, activation='relu', lr=.01, optimizer='adam', beta1=.9, beta2=.999):
        self.lr = lr  # Set the learning rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta1_t = 1.  # = pow(beta1, t) in the t-th iteration/call of compute_grad()
        self.beta2_t = 1.
        self.optimizer = optimizer

        # Set the activation function
        act_map = {'relu': _relu, 'sigmoid': _sigmoid, None: _identity}
        if activation not in act_map:
            ValueError('Activation function', activation, 'is unknown.')
        self.activate = act_map[activation]

        # Set the derivative thereof
        act_der_map = {'relu': _relu_der, 'sigmoid': _sigmoid_der, None: _identity}
        self.activate_der = act_der_map[activation]

        self.__init_weights(n, n_prev, activation)
        self.grad_W = None
        self.grad_b = None

        # Init the Adam GD parameters
        self.W_m = np.zeros((n, n_prev))
        self.W_v = np.zeros((n, n_prev))
        self.b_m = np.zeros((n, 1))
        self.b_v = np.zeros((n, 1))

    # Propagate the input data through this layer & return the result
    # Propagation of a list of samples is possible, where each sample is a column
    def __call__(self, x):
        # x = normalize(x)  # TODO: Normalization would be nice
        self.last_z = self.W @ x + self.b
        self.last_out = self.activate(self.last_z)
        y = self.last_out
        if len(y) == 1:
            return y[0]
        return y

    # Compute the gradient of the loss function w.r.t. the parameters W and b
    # Inspired by https://medium.com/@neuralthreads/backpropagation-made-super-easy-for-you-part-1-6fb4aa5a0aaf
    # ADAM GD, see e.g. p.105 of the lecture notes
    def compute_grad(self, err_grad):
        n_samples = self.last_out.shape[1]

        # Compute the gradient w.r.t. b per sample
        grad_b = err_grad * self.activate_der(self.last_z)

        # Compute the gradient w.r.t. W (averaged over all samples)
        grad_W = np.zeros(self.W.shape)
        for i in range(n_samples):
            grad_W += grad_b[:, i] @ self.last_out[:, i].T
        grad_W /= n_samples

        # Compute the error gradient to propagate
        new_err_grad = self.W.T @ grad_b

        # Compute the gradient w.r.t. b (averaged over all samples)
        grad_b = np.mean(grad_b, axis=1).reshape((-1, 1))

        # Update the Adam parameters
        self.W_m = self.beta1 * self.W_m + (1 - self.beta1) * grad_W
        self.W_v = self.beta2 * self.W_v + (1 - self.beta2) * np.square(grad_W)
        self.b_m = self.beta1 * self.b_m + (1 - self.beta1) * grad_b
        self.b_v = self.beta2 * self.b_v + (1 - self.beta2) * np.square(grad_b)

        return new_err_grad

    # Update the weights W and b using Adam
    def update_weights(self):
        if self.optimizer == 'adam':
            # Compute the relevant Adam parameters
            self.beta1_t *= self.beta1  # beta1_t = pow(beta1, t), where t denotes the t-th iteration
            self.beta2_t *= self.beta2
            W_m_hat = self.W_m / (1 - self.beta1_t)
            W_v_hat = self.W_v / (1 - self.beta2_t)
            b_m_hat = self.b_m / (1 - self.beta1_t)
            b_v_hat = self.b_v / (1 - self.beta2_t)
            eps = 1E-15

            # Update the weights W and b
            self.W -= self.lr / (np.sqrt(W_v_hat) + eps) * W_m_hat
            self.b -= self.lr / (np.sqrt(b_v_hat) + eps) * b_m_hat
        else:
            # Update the weights W and b
            self.W -= self.lr * self.W_m
            self.b -= self.lr * self.b_m
