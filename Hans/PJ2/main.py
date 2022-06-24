import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from nn import NN, loss
from pandas import DataFrame
from progressPlotter import plot_result
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

fname = '../heart.csv'


def read_csv(fname: str):
    X = []
    with open(fname, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            X.append(np.asarray(row, dtype=float))

    return X


# Center the values in X to be centered
def mean_centered(X):
    return X - np.mean(X, axis=0)


# Compute the covariance matrix for the input samples X
def cov_mat(X):
    return X @ X.T / len(X[0])


# Compute the MSE between the given two lists
def mse(y: list, y_true: list):
    return np.mean((y_true - y) ** 2)


# Train the NN
def train(nn: NN, pc: list, y_true: list):
    # Record the starting time to determine the runtime
    t0 = time.time()

    # Init variables
    losses = []
    accuracy = []
    min_loss = float('inf')
    for it in range(N_ITER):
        y_pred = nn(pc.T)  # Feed-forward all samples

        # Loasses
        l = loss(y_pred, y_true)
        losses.append(l)
        if l < min_loss:
            min_loss = l

        # Accuracy
        acc = get_accuracy(y_true, y_pred)
        accuracy.append(acc)

        print('Ep. {:d} | Acc: {:.0f} %, Loss: {:.1f}, Min. loss: {:.1f}'.format(it + 1, acc * 100, l, min_loss))
        nn.learn(y_true)

        # Show the progress/learning
        plot_result(losses, accuracy, cur_it=it + 1, n_iter=N_ITER)
        if float('nan') in y_pred:
            print('WARNING: NaN detected in y_pred.')
            break

    # Record & return the elapsed time
    t = time.time() - t0
    print('Training costed {:.2f} s'.format(t))

    return t


# NICE TODO: Validate (using the validation set)
def validate(nn: NN, x_train: list, batch_size: int):
    mini_batches = sample_as_mini_batches(x_train, batch_size)
    pass


# TODO: Test (using the test set)
def test(nn: NN, X: list, y_true: list):
    y_pred = nn(X.T)
    a = get_accuracy(y_true, y_pred)
    print("Test accuracy {:.2f}".format(a))

    return y_pred, a

def predict(nn: NN, X: list):
    return nn(X)

def make_uniform_grid(X):
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1

    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    xx, yy = np.meshgrid(x1grid, x2grid)

    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))

    return xx, yy, grid


# Split the data into training and validation data
def split_data(X: list, Y: list, test_size: float):
    # Apply stratified sampling
    return train_test_split(X, Y, stratify=Y, test_size=test_size)


# Convert the given pdf to the cdf
def pdf2cdf(pdf: list):
    return np.pad(np.cumsum(pdf), (1, 0), mode='constant')


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
def plot(pc, Y, title: str):
    Y = np.array(Y)
    pc1 = pc[Y == 1, :]
    pc2 = pc[Y != 1, :]
    plt.plot(pc1[:, 0], pc1[:, 1], '.')
    plt.plot(pc2[:, 0], pc2[:, 1], '.')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(['Sick', 'Healthy'])
    plt.show()


# Determine the accuracy, i.e. the percentage of labels predicted correctly
def get_accuracy(y_true, y_pred):
    return sum(y_true == (y_pred > .5)) / len(y_true)


def sample_as_mini_batches(x_train: np.ndarray, batch_size: int):
    random_samples = np.array(random.sample(set(range(len(x_train))), len(x_train)))
    random_samples = np.array_split(random_samples, batch_size)

    assert sum([len(batch) for batch in random_samples]) == len(
        x_train), "some samples in the training set are not added"
    return [x_train[arr] for arr in random_samples]


# @jit(nopython=True, parallel=True)
def kMeans(X: np.ndarray, y_true, K: int):
    # log('Starting K-means clustering using the metric \'%s\'' % metric)

    # Randomly initialize the centroids c
    N, M = X.shape
    c = X[np.random.randint(N, size=K)]
    y = [-1] * N  # Labels

    ePrev = M * N + 1
    eNew = ePrev - 1
    while ePrev - eNew > 0:
        ePrev = eNew
        eNew = 0

        # Assign x_i to cluster k = argmin_j || x_i - c_j ||
        for i, x in enumerate(X):
            dists = cdist([x], c, 'sqeuclidean')
            y[i] = np.argmin(dists)

        # Update the centroid positions
        for k in range(K):
            Ck = X[[i for i, yi in enumerate(y) if yi == k]]  # Compute Ck = {x_i where the label y_i = k}
            if len(Ck) > 0:
                c[k] = Ck.mean(axis=0)  # Centroid = the mean of the elements/columns in Ck
                eNew += cdist([c[k]], Ck).sum()  # Compute the error
            else:
                # Try to find a good position for this centroid
                c[k] = c.mean()

    # Determine the accuracy
    a = 0
    for k in range(K):
        Lk = y_true[[i for i, yi in enumerate(y) if yi == k]]  # Lk = {y_true_i : y_i = k}
        Lk = DataFrame(data=Lk)
        cnt = Lk.value_counts()
        cnt = cnt.values
        if len(cnt) > 0:
            a += cnt[0]
    a /= N

    return y, a  # Return the labels & accuracy


# TODO: Craete a plot showing the scattered data & labeling areas
def create_plot_data(nn: NN, y_pred: list):
    pass


if __name__ == '__main__':
    N_ITER = 50

    # Read the data
    X = read_csv(fname)

    # Split the data in to the samples and labels
    X = np.array(X)
    Y = X[:, 13]  # Labels
    X = X[:, :13]
    print('X.shape:', X.shape)

    # Make the labels binary, i.e. (+1, -1) -> (1, 0)
    Y = (np.array(Y) + 1) / 2

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
    plot(pc, Y, 'Reduced Feature Space')

    # Apply K-Means to see its performance
    K = 2
    y_pred, a = kMeans(pc, Y, K)
    print('K-Means K={:d} gives an acc. of {:.0f} %'.format(K, a * 100))

    # plot(pc, y, 'K-Means K={:d}'.format(K))  # TODO (Sattish): use this line & plot the labeling areas in the background
    plot(pc, y_pred, 'K-Means K={:d}'.format(K))

    lr_set = [.2, .1, .05]
    Ki_set = [2, 5, 10]
    N_set = [2, 5, 10]

    for lr in lr_set:
        for Ki in Ki_set:  # #neurons per layer
            for N in N_set:  # #leyare
                print('Start training using {:d} layers, each containing {:d} neurons, with lr = {:.3f}'.format(N, Ki, lr))

                # Create a NN
                n_per_layer = [Ki for i in range(N)]
                nn = NN(2, 1, N, n_per_layer, lr=lr)

                X_train, X_test, y_train, y_test = split_data(pc, Y, 0.25)  # Take 25% of the data set as test data

                train(nn, X_train, y_train)

                ### contour plot to distinguish the two classes in 2D feature space
                xx, yy, uniformPC = make_uniform_grid(pc)
                y_pred = predict(nn, uniformPC.T)
                zz = y_pred.reshape(xx.shape)
                plt.contourf(xx, yy, zz, cmap = 'Paired');
                plt.show()
                
                # validate(nn, x_val, y_val)
                y_pred, a = test(nn, X_test, y_test)
                # plots.append(create_plot_data(nn, y_pred))  # TODO: this is just an idea for the 3x3 plot

        # TODO: For each learning rate lr, create a plot of the 9 results (1 for each (Ki, N) combi)
        # plot_all(plots)  # TODO: this is just an idea for the 3x3 plot