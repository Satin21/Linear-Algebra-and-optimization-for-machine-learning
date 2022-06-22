import argparse
import csv
import time

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
    # Record the starting time to determine the runtime
    t0 = time.time()

    # Init variables
    losses = []
    min_loss = float('inf')
    n = len(pc)
    for ep in range(N_EPOCHS):

        # Select the i-th sample
        i = np.random.randint(n)
        x = pc[i]
        y_true = y[i]

        # TODO: Use the whole data set
        y_hat = nn(x.T)  # Feed-forward
        # y_hat = nn(pc.T)  # Feed-forward all samples
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
def validate(nn: NN, X: list, y_true: list):
    y_pred = nn(X)
    a = get_accuracy(y_true, y_pred)
    print("Validation accuracy {:.2f}".format(a))


# TODO: Test (using the test set)
def test(nn: NN, X: list, y_true: list):
    y_pred = nn(X)
    a = get_accuracy(y_true, y_pred)
    print("Test accuracy {:.2f}".format(a))


# Split the data into training and validation data
def split_data(X: list, Y: list, test_size: float):

    # Apply stratified sampling
    return train_test_split(X, Y, stratify=Y, test_size=test_size)


def pdf2cdf(x: list):
    return np.pad(np.cumsum(x), (1, 0), mode='constant')


# Plot the CDF
def plot_cdf(X: list):
    # Do the PCA
    n_features = len(X[0])
    pca = PCA(n_components=n_features)
    pca.fit_transform(X)  # Compute the principal components & transform the data
    cdf = pdf2cdf(pca.explained_variance_ratio_)

    plt.plot(cdf, '.-')
    plt.title('Explained Variance CDF')
    plt.xlabel('#Principal Components')
    plt.ylabel('CDF')
    plt.grid()
    plt.show()


# Plot the transformed data set X, i.e. the principal components
def plot_x_transformed(pc, Y):
    pc1 = pc[Y == 1, :]
    pc2 = pc[Y == -1, :]
    plt.plot(pc1[:, 0], pc1[:, 1], '.')
    plt.plot(pc2[:, 0], pc2[:, 1], '.')
    plt.title('Reduced Feature Space')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(['Sick', 'Healthy'])
    plt.show()


# Determine the accuracy, i.e. the percentage of labels predicted correctly
def get_accuracy(y_true, y_pred):
    a = 0
    for k in range(K):
        Lk = y_true[[i for i, yi in enumerate(y_pred) if yi == k]]  # Lk = {y_true_i : y_i = k}
        Lk = DataFrame(data=Lk)
        cnt = Lk.value_counts()
        cnt = cnt.values
        if len(cnt) > 0:
            a += cnt[0]
    return a / N


if __name__ == '__main__':
    N_EPOCHS = 500

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

    # plot_cdf(X)
    # plot_x_transformed(pc, Y)

    lr_set = [.11, .05, .01]
    Ki_set = [2, 5, 10]
    N_set = [2, 5, 10]

    for lr in lr_set:
        for Ki in Ki_set:  # #neurons per layer
            for N in N_set:  # #leyare

                # Create a NN
                n_per_layer = [Ki for i in range(N)]
                nn = NN(2, 1, N, n_per_layer, lr=lr)

                x_val, y_val, x_test, y_test = split_data(X, Y, 0.75)

                train(nn, pc, Y)
                # validate(nn, x_val, y_val)
                test(nn, x_test, y_test)
                break
            break
        break
