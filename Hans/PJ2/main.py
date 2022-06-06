import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from nn import NN
from progressPlotter import plot_result
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

fname = '../heart.csv'


def read_csv(fname):
    X = []
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            # print(', '.join(row))
            X.append(np.asarray(row, dtype=float))

    return X


def mean_centered(X):
    return X - np.mean(X, axis=0)


def cov_mat(X):
    return X @ X.T / len(X[0])


def mse(y, y_true):  # MSE
    return np.mean((y_true - y) ** 2)


# Define the loss function
def loss(y, y_true):
    return mse(y, y_true)


# Train the NN
def train(nn, pc, y, n_epochs):
    in_vals = np.zeros((n_epochs, 2))
    losses = []
    min_loss = float('inf')
    best_in = None
    n = len(pc)
    for ep in range(n_epochs):
        # x = pc[0:4]
        # y_true = y[0:4]
        # Select the i-th sample
        i = np.random.randint(n)
        x = pc[i]
        y_true = y[i]

        y_hat = nn(x.T)  # Feed-forward

        in_vals[ep] = x  # Record the values
        l = loss(y_hat, y_true)
        losses.append(l)
        if l < min_loss:
            min_loss = l
            best_in = x

        print(f'Ep. {ep} | loss:', loss(y_hat, y_true))
        nn.learn(y_true)

        # Show the progress/learning
        plot_result(in_vals, best_in[0], losses, ep + 1, 1, N_EPOCHS)


if __name__ == '__main__':
    # Get the program parameters
    parser = argparse.ArgumentParser(
        prog="Project 2 | Neural Network Simulation",
        description="A neural network is implemented & tested on the data in heart.csv for different network topologies."
    )
    # parser.add_argument('--sigma2', type=float, nargs='?', default=.1,
    #                     help="The variance of the noise used by exploration.")
    parser.add_argument('--n_ep', type=int, nargs='?', default=100,
                        help="The number of epochs.")
    parser.add_argument('--lr', type=float, nargs='?', default=5e-3,
                        help="The learning rate.")
    # parser.add_argument('--tau', type=float, nargs='?', default=0.01,
    #                     help="Weights update parameter.")
    # parser.add_argument('--batch_size', type=int, nargs='?', default=64,
    #                     help="Batch size.")

    args = parser.parse_args()
    # SIGMA2 = args.sigma2  # Noise variance in the received signals
    N_EPOCHS = args.n_ep
    LR = args.lr
    # TAU = args.tau
    # BATCH_SIZE = args.batch_size

    # Read the data
    X = read_csv(fname)

    # Split the data in to the samples and labels
    X = np.array(X)
    y = X[:, 13]  # Labels
    X = X[:, :13]
    print('X.shape:', X.shape)

    X = mean_centered(X)
    Cxx = cov_mat(X)
    print('Cxx.shape:', Cxx.shape)

    # Scale the input data/features
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Do the PCA
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X)  # Compute the principal components & transform the data
    print('explained_variance:', pca.explained_variance_ratio_)
    print(pc.shape)

    # Plot the transformed data
    pc1 = pc[y == 1, :]
    pc2 = pc[y == -1, :]
    plt.plot(pc1[:, 0], pc1[:, 1], '.')
    plt.plot(pc2[:, 0], pc2[:, 1], '.')
    plt.title('Reduced Feature Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    # plt.show()

    # Create a NN
    n_layers = 2
    Ki = 2
    n_per_layer = [Ki for i in range(n_layers)]
    nn = NN(2, 1, n_layers, n_per_layer, LR)

    train(nn, pc, y, N_EPOCHS)