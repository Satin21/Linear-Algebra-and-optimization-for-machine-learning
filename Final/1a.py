"""
Script that implements that k-means algorithm on the MNIST dataset, All necessary functions for the script 
are defined below.
"""

# Import the dependencies
import time
from math import atan, degrees
import cv2 as cv
from numpy import (
    argmin,
    argmax,
    array,
    empty,
    exp,
    ma,
    ndarray,
    nonzero,
    histogram,
    random,
    zeros,
)
from numpy.linalg import svd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans  # to verify the performance
from sklearn.datasets import fetch_openml
from sklearn.metrics import homogeneity_score, completeness_score, accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations
import random

# Set TRUE for the final version to deliver!
final = True


# Some help functions
def log(msg):
    t = time.localtime()
    curr_time = time.strftime("%H:%M:%S", t)
    print(curr_time, msg)


# Give the homogeneity, completeness & accuracy score for the given labeling
def scores(true_labels, labels):
    hom = homogeneity_score(true_labels, labels)
    com = completeness_score(true_labels, labels)
    acc = accuracy_score(true_labels, labels)

    return hom, com, acc


def cluster_initialization(X, nclusters):

    """
    Initialize the clusters by maximizing the Sqeuclidean distance between the centroids

    Parameters:
    ----------
    X  (np.ndarray)  data
    nclusters (int) no of clusters


    """

    distance = []
    positions = []
    for i in range(nclusters * nclusters):
        random_positions = random.sample(range(len(X)), nclusters * nclusters)
        combos = []
        for i in range(nclusters):
            combos.append(
                sum(
                    [
                        np.sum((X[indices[0]] - X[indices[1]]) ** 2)
                        for indices in list(
                            combinations(
                                random_positions[nclusters * i : nclusters * (i + 1)], 2
                            )
                        )
                    ]
                )
            )
            max_index = np.argmax(combos)
            positions.append(
                random_positions[nclusters * max_index : nclusters * (max_index + 1)]
            )
            distance.append(combos[np.argmax(combos)])

    return positions, distance


def kMeans(X: ndarray, y_true, K: int, metric: str):
    # log('Starting K-means clustering using the metric \'%s\'' % metric)

    # Initialize the centroids c
    positions, max_distance = cluster_initialization(X, K)
    c = X[positions[np.argmax(max_distance)]]
    y = [-1] * N  # Labels

    ePrev = M * N + 1
    eNew = ePrev - 1
    while ePrev - eNew > 0:
        ePrev = eNew
        eNew = 0

        # Assign x_i to cluster k = argmin_j || x_i - c_j ||
        for i, x in enumerate(X):
            if metric == "gauss":
                gamma = 0.001
                dists = cdist([x], c, "sqeuclidean")
                dists = exp(-gamma * dists)
            else:
                dists = cdist([x], c, metric)
            y[i] = argmin(dists)

        # Update the centroid positions
        for k in range(K):
            Ck = X[
                [i for i, yi in enumerate(y) if yi == k]
            ]  # Compute Ck = {x_i where the label y_i = k}
            if len(Ck) > 0:
                c[k] = Ck.mean(
                    axis=0
                )  # Centroid = the mean of the elements/columns in Ck
                eNew += cdist([c[k]], Ck).sum()  # Compute the error
            else:
                # Try to find a good position for this centroid
                c[k] = c.mean()

    # Determine the accuracy
    a = 0
    for k in range(K):
        Lk = y_true[
            [i for i, yi in enumerate(y) if yi == k]
        ]  # Lk = {y_true_i : y_i = k}
        Lk = DataFrame(data=Lk)
        cnt = Lk.value_counts()
        cnt = cnt.values
        if len(cnt) > 0:
            a += cnt[0]
    a /= N

    #     showCentroids(c)

    return y, a  # Return the labels


def SVD(data, k=1e-1, L=None, return_compressed=True, return_L=False):

    """
    Compute the SVD decomposition

    Parameters:
    ------
    data    (np.ndarray)           2D array representing an image
    k       (float)                threshold to choose the singular values
    L       (int)                  no. of singular values (degree of compression)
    return_compressed (bool)       Whether to return the compressed image
    return_L (bool)                Whether to return the no. of singular values

    Returns:
    _______

    Returns the compressed image (np.ndarray) as 2D array (or) returns the no. of singular values based on the parameters.


    """

    U, s, V = np.linalg.svd(data, full_matrices=True)
    if L == None:
        L = sum(s > k)
    if return_compressed:
        return U[:, :L] @ np.diag(s)[:L, :L] @ V[:L, :]
    if return_L:
        return L


# Visualize the given centroids
def showCentroids(c):
    print("Centroids:")
    plt.figure(figsize=(20, 10))
    if isinstance(c, DataFrame):
        c = c.to_numpy()
    for i, x in enumerate(c):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x.reshape(28, 28))
        plt.axis("off")
    plt.show()


# Show 10x zero, 10x one, ..., 10x nine
def show10(X, y):
    K = 10

    # For each digit 0, ..., 9, retrieve 10 samples
    plt.figure(figsize=(10, 10))
    n = 1
    for k in range(K):
        Xk = X.loc[y == k].sample(K)
        for i, x in Xk.iterrows():
            plt.subplot(10, 10, n)
            plt.imshow(x.values.reshape(28, 28))
            plt.axis("off")
            n += 1
    plt.show()


# Load the data
print("Loading the data")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, cache=True)
print("The data are loaded")

# For testing: Split the data set into a smaller training (& validation) set
if final:
    X_train = X
    y_train = y
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.9)
log("Let us train using " + str(len(X_train)) + " samples.")

# Remove the original indices & scale X
X_train = X_train.reset_index(drop=True) / 255
y_train = y_train.astype("int").reset_index(drop=True)

# Just to get an impression of the data contents (labels)
print("Frequency per label:\n", y_train.value_counts())

# Convert the data to arrays
# X_train = X_train.to_numpy()
y_train = y_train.to_numpy()


def doCluster(X: ndarray, y, show=True):
    K = 10

    def A(kmeans):
        c = kmeans.cluster_centers_
        y_found = kmeans.labels_
        s = 0
        for k in range(K):
            df = DataFrame(data=y[y_found == k])
            cnt = df.value_counts()
            s += cnt.values[0]
        return s / len(X)

    # Apply the K-means clustering algorithm using the square Euclidean distance measure
    start = time.time()
    #     XX = X.to_numpy
    myLabels, a1 = kMeans(X, y, K, "sqeuclidean")
    t1 = time.time() - start
    myScore = scores(y, myLabels)

    # Use correlation as distance measure
    start = time.time()
    myLabels2, a2 = kMeans(X, y, K, "correlation")
    t2 = time.time() - start
    myScore2 = scores(y, myLabels2)

    # Use the Gaussian distance measure
    start = time.time()
    myLabels3, a3 = kMeans(X, y, K, "gauss")
    t3 = time.time() - start
    myScore3 = scores(y, myLabels3)

    # Use an existing algorithm
    start = time.time()
    #     kmeans = KMeans(n_clusters=K, random_state=1).fit(X)
    kmeans = KMeans(n_clusters=K, init="k-means++").fit(X)  # Slight speedup
    t = time.time() - start
    score = scores(y, kmeans.labels_)

    # Print the results
    log(
        "Results:\nAlgorithm \tHomogeneity Completeness   Accuracy   Time [s]\n"
        "My K-Means \t%11.4f %12.4f %10.1f%% %9.3f"
        % (myScore[0], myScore[1], a1 * 100, t1)
        + "\nMy K-Means corr.%11.4f %12.4f %10.1f%% %9.3f"
        % (myScore2[0], myScore2[1], a2 * 100, t2)
        + "\nMy K-Means gauss.%10.4f %12.4f %10.1f%% %9.3f"
        % (myScore3[0], myScore3[1], a3 * 100, t3)
        + "\nExist. K-Means\t%11.4f %12.4f %10.1f%% %9.3f"
        % (score[0], score[1], A(kmeans) * 100, t)
    )

    # if show:
    #     showCentroids(kmeans.cluster_centers_)

    return [a1, a2, a3, A(kmeans)], [t1, t2, t3, t]


X_train_arr = X_train.to_numpy()


# doCluster(X_train_arr, y_train)


# Equalize the histogram
def equalize(img):
    hist, bins = histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = ma.filled(cdf_m, 0).astype("uint8")

    return cdf[img]


# Given a set of points with coordinates (x,y) in X x Y, find the best line fit y = a + b * x & return a, b.
# Based on code from
# https://stackoverflow.com/questions/22239691/code-for-best-fit-straight-line-of-a-scatter-plot-in-python
def fitLine(X, Y):
    xbar = sum(X) / len(X)
    ybar = sum(Y) / len(Y)
    n = len(X)

    numer = sum([xi * yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    #     print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b


# Rotate the given image by [angle] degrees
# Inspired by https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
def rotate(img, angle):
    #     print('rotate {:.2f} deg'.format(angle))
    img_center = tuple(array(img.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(img_center, angle, 1.0)
    return cv.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv.INTER_LINEAR)


# Detect & correct the rotation of the written digit
def correctRotation(img):
    nz = nonzero(img)  # Find the mean index of the nonzeros per row
    df = DataFrame(data=nz)
    means = df.T.groupby(0).mean()
    inds = means.index.to_numpy()  # Make it an ndarray
    _, b = fitLine(
        inds, means[1].values
    )  # Fit a line (y = a + b * x) in order to detect the rotation
    return rotate(img, degrees(atan(-b)))  # Rotate the image


# Sharpen the given image
def sharpen(img):
    sharpen_filter = array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv.filter2D(img, -1, sharpen_filter)

    return img


# Define an image processing function. Use r as radius for the Gaussian blurring step
def procImg(img, r):
    img = (255 * img).astype("uint8")  # Convert the image to uint8 to process it
    img = cv.GaussianBlur(img, (r, r), 0)  # Blur
    img = cv.equalizeHist(img)  # Equalize
    # Other tries were the following, though they did not improve the clustering accuracy.
    #     img = equalize(img)
    #     img = sharpen(img)                     # Sharpen
    #     img = setBounds(img)
    #     img[img < 80] = 0  # Threshold
    #         # img = cv.Canny(img, 30, 150)    # Edge detection
    # Sum each row and put the value in the right most pixel
    #         img[:,27] = sum(img, 2)
    #         img[27,:] = sum(img, 1)
    img = correctRotation(img)
    #     img = correctRotation(img)  # Try a second correction

    return img.astype("float32") / 255


# Test the image processor
# img = X_train.loc[10].values.reshape(28, 28)
# imgP = procImg(img, 3)
#
# plt.subplot(1,2,1)
# plt.imshow(img)
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(imgP)
# plt.axis('off')
# plt.show;


# Process the images. Use r as radius for the Gaussian blurring step.
def process(X: ndarray, r):
    X_proc = X.copy()
    for i, x in enumerate(X):
        # Process image x
        img = x.reshape(28, 28)
        img = procImg(img, r)
        X_proc[i] = img.reshape(784)

    return X_proc


# Test the performance (resulting accuracy) of applying K-means after a blurring step
K = 10
r_max = 7
r_vals = range(1, r_max + 1, 2)
a = [0] * len(r_vals)  # Accuracy
t = [0] * len(r_vals)  # Processing times
for i in range(len(r_vals)):
    r = r_vals[i]
    log("Test r = {:d}".format(r))

    # Process the images using blurring radius r
    t_i = time.time()
    X_proc = process(X_train_arr, r)
    t[i] = time.time() - t_i

    # Cluster the processed data using K-Means & record the accuracy (mean over several runs)
    n_acc = 3
    acc = 0
    for j in range(n_acc):
        _, acc_j = kMeans(X_proc, y_train, K, "sqeuclidean")
        acc += acc_j
    a[i] = acc / n_acc

# Plot the results
print(len(X_train), "samples were used.")


# print('The average image processing time for this data set was', array(t).mean(), 's.')


def plotAccTime(x_vals, x_label, a, t, title):
    # Plot the accuracy
    fig, ax = plt.subplots()
    ax.plot(x_vals, 100 * array(a), ".-")
    ax.tick_params(axis="y", labelcolor="blue")
    ax.legend(["Accuracy"])

    # Plot the image processing time
    ax2 = ax.twinx()
    ax2.plot(x_vals, t, ".-k")
    ax2.legend(["Processing Time"], loc="lower right")
    plt.xlabel(x_label)
    plt.ylabel("Accuracy [%]")
    plt.title(title)


plotAccTime(r_vals, "Radius r", a, t, "Blurring Efficiency")

# Select the best radius r
r_best = r_vals[argmax(a)]
print(
    "The best radius is",
    r_best,
    ", giving an accuracy of {:2.1f} %.\nThe resulting processing time is approx. {:.0f} s.".format(
        100 * max(a), t[argmax(a)]
    ),
)

# Reprocess the data using the best radius r_best
X_proc = process(X_train_arr, r_best)


# Apply the clustering algorithm on the processed images
doCluster(X_proc, y_train)

# Compress the images using SVD
compressed = np.array(
    list(map(lambda x: SVD(x.reshape(28, -1), L=3).flatten(), X_train_arr))
)
