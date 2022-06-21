import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from nn import NN
from progressPlotter import plot_result
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
def train(nn: NN, pc: list, y: list):
    # TODO: time it

    losses = []
    min_loss = float('inf')
    n = len(pc)
    for ep in range(N_EPOCHS):
        # x = pc[0:4]
        # y_true = y[0:4]
        # Select the i-th sample
        i = np.random.randint(n)
        x = pc[i]
        y_true = y[i]

        # TODO: Use the whole data set
        y_hat = nn(x.T)  # Feed-forward
        # y_hat = nn(pc.T)  # Feed-forward
        # print(y_hat)
        l = loss(y_hat, y_true)
        losses.append(l)
        if l < min_loss:
            min_loss = l

        print(f'Ep. {ep} | loss:', l)
        nn.learn(y_true)

        # Show the progress/learning
        plot_result(rewards=losses, cur_it=ep + 1, n_iter=N_EPOCHS)


# TODO: Validate (using the validation set)
def validate(nn: NN, X: list, Y: list):
    pass


# TODO: Test (using the test set)
def test(nn: NN, X: list, Y: list):
    pass


# Split the data into training and validation data
def split_data(X: list, Y: list, test_size: float):

    # Apply stratified sampling
    return train_test_split(X, Y, stratify=Y, test_size=test_size)


def pdf2cdf(x):
    return np.pad(np.cumsum(x), (1, 0), mode='constant')


if __name__ == '__main__':
    # Get the program parameters
    parser = argparse.ArgumentParser(
        prog="Project 2 | Neural Network Simulation",
        description="A neural network is implemented & tested on the data in heart.csv for different network topologies."
    )
    # parser.add_argument('--sigma2', type=float, nargs='?', default=.1,
    #                     help="The variance of the noise used by exploration.")
    parser.add_argument('--n_ep', type=int, nargs='?', default=200,
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
    Y = X[:, 13]  # Labels
    X = X[:, :13]
    print('X.shape:', X.shape)

    X = mean_centered(X)
    # Cxx = cov_mat(X)
    # print('Cxx.shape:', Cxx.shape)

    # Scale the input data/features
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Do the PCA
    pca = PCA(n_components=2)
    pc = pca.fit_transform(X)  # Compute the principal components & transform the data
    print('explained variance:', pca.explained_variance_)
    print('explained variance ratio:', pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

    # CDF
    cdf = pdf2cdf(pca.explained_variance_ratio_)

    # Plot the CDF
    # plt.plot(cdf, '.-')
    plt.title('Explained Variance CDF')
    plt.xlabel('#Principal Components')
    plt.ylabel('CDF')
    plt.grid()
    # plt.show()
    print(pc.shape)

    # Plot the transformed data
    pc1 = pc[Y == 1, :]
    pc2 = pc[Y == -1, :]
    plt.plot(pc1[:, 0], pc1[:, 1], '.')
    plt.plot(pc2[:, 0], pc2[:, 1], '.')
    plt.title('Reduced Feature Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(['Sick', 'Healthy'])
    #plt.show()

    # Create a NN
    n_layers = 2  # #layers
    Ki = 2  # #neurons per layer
    n_per_layer = [Ki for i in range(n_layers)]
    nn = NN(2, 1, n_layers, n_per_layer, LR)

    x_val, y_val, x_test, y_test = split_data(X, Y, 0.75)

    train(nn, pc, Y)
    validate(nn, x_val, y_val)
    test(nn, x_test, y_test)
