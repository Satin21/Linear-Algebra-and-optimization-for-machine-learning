import argparse
import csv
import time

import matplotlib.pyplot as plt
import numpy as np
from KMeans import KMeans
from matplotlib.colors import ListedColormap
from nn import NN, loss
from progressPlotter import plot_result
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


# Get a batch of size batch_size of the given data samples in X
def get_batch(X: list, y_true: list, batch_size=16):
    batch = np.random.choice(len(X), batch_size, replace=False)
    return X[batch], y_true[batch]


# Train the NN
def train(nn: NN, pc: list, y_true: list):
    # Record the starting time to determine the runtime
    t0 = time.time()
    t = 0

    # Init variables
    losses = []
    accuracy = []
    min_loss = float('inf')
    best_acc = 0
    for it in range(N_ITER):

        # Select a batch to do the NN optimization with
        batch_size = len(pc)
        X_batch, y_batch = get_batch(pc, y_true, batch_size)

        # Do the forward propagation using the batch
        y_pred = nn(X_batch.T)

        # Compute the loss
        l = loss(y_pred, y_batch)
        losses.append(l)
        if l < min_loss:
            min_loss = l

        # Accuracy
        acc = get_accuracy(y_batch, y_pred)
        accuracy.append(acc)
        if acc > best_acc:
            best_acc = acc

        print('Ep. {:d} | Acc: {:.1f} %, Best acc.: {:.1f} %. Loss: {:.1f}, Min. loss: {:.1f}'.format(it + 1, acc * 100,
                                                                                                      best_acc * 100, l,
                                                                                                      min_loss))

        # Do the backward propagation
        nn.learn(y_batch)

        if it == N_ITER - 1:
            # Record & return the elapsed time
            t = time.time() - t0

        # Show the progress/learning
        plot_result(losses, accuracy, cur_it=it + 1, n_iter=N_ITER, fname='loss_acc.png')
        if float('nan') in y_pred:
            print('WARNING: NaN detected in y_pred.')
            break

    if t == 0:
        # Record & return the elapsed time
        t = time.time() - t0
    print('Training costed {:.2f} s'.format(t))

    return t


# NICE TODO: Validate (using the validation set)
def validate(nn: NN, x_train: list, batch_size: int):
    mini_batches = sample_as_mini_batches(x_train, batch_size)
    pass


# Test the preformance (using the test set)
def test(nn: NN, X: list, y_true: list):
    y_pred = nn(X.T)
    a = get_accuracy(y_true, y_pred)
    print("Test accuracy: {:.2f}".format(a))

    return y_pred, a


def predict(nn: NN, X: list):
    return nn(X)


def make_uniform_grid(X):
    min1, max1 = X[:, 0].min() - .5, X[:, 0].max() + .5
    min2, max2 = X[:, 1].min() - .5, X[:, 1].max() + .5

    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)

    xx, yy = np.meshgrid(x1grid, x2grid)

    # Flatten each grid to a vector & stack them
    return xx, yy, np.c_[xx.ravel(), yy.ravel()]


# Split the data into training and test data
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


# Create a scatter plot showing the samples & draw a contour plot to distinguish the two classes in 2D feature space
def plot_boundaries_and_scatter(X: list, y_true: np.ndarray, nn: NN = None, fname=None):
    # Calculate the mesh data
    xx, yy, uniformPC = make_uniform_grid(X)
    if nn is not None:
        zz = predict(nn, uniformPC.T).reshape(xx.shape)
        zz[zz < .5] = 0
        zz[zz >= .5] = 1.
        title = 'NN Decision Boundaries'
    else:
        zz = kMeans.predict(uniformPC).reshape(xx.shape)
        title = 'K-Means K={:d} Decision Boundaries'.format(K)

    # Create color maps
    cnt0 = np.count_nonzero(zz)
    if cnt0 == np.prod(zz.shape):
        cmap_light = ListedColormap(['#FFAAAA'])
    else:
        cmap_light = ListedColormap(['#AAFFAA', '#FFAAAA'])
    cmap_bold = ListedColormap(['#00FF00', '#FF0000'])

    # Draw the mesh data & data samples
    plt.contourf(xx, yy, zz, cmap=cmap_light)
    plt.scatter(X[y_true == 0, 0], X[y_true == 0, 1], c=cmap_bold.colors[0], label='Healthy')
    plt.scatter(X[y_true == 1, 0], X[y_true == 1, 1], c=cmap_bold.colors[1], label='Sick')

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    fig = plt.gcf()
    plt.show()
    if fname is not None:
        plt.draw()
        fig.savefig(fname, dpi=100, bbox_inches='tight')


if __name__ == '__main__':
    N_ITER = 500

    # Read the data
    X = read_csv(fname)

    # Split the data in to the samples and labels
    X = np.array(X)
    Y = X[:, 13]  # Labels
    X = X[:, :13]  # Features
    print('Data from {:d} patients are read'.format(len(Y)))

    # Make the labels binary, i.e. (+1, -1) -> (1, 0)
    Y = (np.array(Y) + 1) / 2
    print('{:.1f} % of the patients is healthy'.format(100 * sum(Y == 0) / len(Y)))

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
    # plot(pc, Y, 'Reduced Feature Space')

    # Apply K-Means to see its performance
    K = 2
    n_runs = 10
    kMeans = KMeans()
    a_avg = 0
    for i in range(n_runs):
        y_pred, a = kMeans(pc, Y, K)
        a_avg += a
    a_avg /= n_runs
    print('K-Means cluster centers:', kMeans.cluster_centers)
    print('K-Means K={:d} gives an avg. acc. of {:.1f} % over {:d} runs'.format(K, a_avg * 100, n_runs))
    plot_boundaries_and_scatter(pc, Y)

    # lr=.01, Ki=10, N=5 gives good results
    # and (.001, 10, 10)
    # lr_set = [.02, .01, .002]
    lr_set = [.002]
    # Ki_set = [2, 5, 10]
    Ki_set = [10]
    # N_set = [2, 5, 10]
    N_set = [10]

    for lr in lr_set:
        for Ki in Ki_set:  # #neurons per layer
            for N in N_set:  # #leyare
                print('Start training using {:d} layers, each containing {:d} neurons, with lr = {:.3f}'.format(N, Ki,
                                                                                                                lr))

                # Create a NN
                n_per_layer = [Ki for i in range(N)]
                nn = NN(2, 1, N, n_per_layer, lr=lr)  # Train using the Adam gradient optimizer
                # nn = NN(2, 1, N, n_per_layer, lr=lr, optimizer='gd')  # Train using standard gradient descent

                X_train, X_test, y_train, y_test = split_data(pc, Y, 0.25)  # Take 25% of the data set as test data

                train(nn, X_train, y_train)

                # Contour plot to distinguish the two classes in 2D feature space
                fname = 'nn_db.png'
                plot_boundaries_and_scatter(pc, Y, nn, fname=fname)

                # validate(nn, x_val, y_val)
                y_pred, a = test(nn, X_test, y_test)
                # plots.append(create_plot_data(nn, y_pred))  # TODO: this is just an idea for the 3x3 plot

        # TODO: For each learning rate lr, create a plot of the 9 results (1 for each (Ki, N) combi)
        # plot_all(plots)  # TODO: this is just an idea for the 3x3 plot
