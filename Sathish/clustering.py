import numpy as np
from itertools import combinations
import random


def polynomial_kernel(x1, x2):
    return (1 + x1.T * x2 ) ** 2

def find_accuracy(nclusters, ytrue, cluster_indices):
    digit = np.zeros((nclusters, ), dtype = int)
    count = np.zeros((nclusters, ), dtype = int)
    cardinality = np.zeros((nclusters, ), dtype = int)
    for i in range(nclusters):
        unique_digits = np.unique(ytrue[cluster_indices == i], return_counts = True)
        digit[i] = unique_digits[0][np.argmax(unique_digits[1])]
        count[i] = unique_digits[1][np.argmax(unique_digits[1])]
        cardinality[i] = sum(unique_digits[1])

    accuracy = (sum(count)/ sum(cardinality)) * 100 # %
    return accuracy

def SVD(data, k = 1e-1, L = None, return_compressed = True, return_L = False):

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


def cluster_initialization(X, nclusters):

    max_distance = []
    positions = []
    for i in range(nclusters * nclusters):
        random_positions = random.sample(range(len(X)), nclusters * nclusters)
        combos = []
        for i in range(10):
            combos.append(
                sum(
                    [
                        np.sum((X[indices[0]] - X[indices[1]]) ** 2)
                        for indices in list(
                            combinations(random_positions[nclusters * i : nclusters * (i + 1)], 2)
                        )
                    ]
                )
            )
            max_index = np.argmax(combos)
            positions.append(random_positions[nclusters * max_index : nclusters * (max_index + 1)])
            max_distance.append(combos[np.argmax(combos)])

    return positions, max_distance



def K_means_clustering(X, n_clusters, subset, max_iter= np.inf, tolerance=1e-16):

    positions, max_distance = cluster_initialization(X, n_clusters)

    centroids = []
    clusters = {}
    for i, index in enumerate(positions[np.argmax(max_distance)]):
        centroids.append(X[index])
        clusters[i] = [X[index]]

    data_distances_to_centroids = np.zeros((subset, 10))

    for cluster_index in list(clusters.keys()):
        data_distances_to_centroids[:, cluster_index] = np.sum(
            (X[:subset] - centroids[cluster_index]) ** 2, axis=1
        )

    old_residual = np.sum(data_distances_to_centroids)
    residual = old_residual + tolerance * 2
    no_iterations = 0

    while np.abs(residual - old_residual) > tolerance and (no_iterations < max_iter):

        min_distance = np.argmin(data_distances_to_centroids, axis=1)

        for i in list(clusters.keys()):
            clusters[i] = X[min_distance == i]

        centroids = np.array(
            [np.mean(cluster, axis=0) for _, cluster in clusters.items()]
        )

        for cluster_index in list(clusters.keys()):
            data_distances_to_centroids[:, cluster_index] = np.sum(
                (X[:subset] - centroids[cluster_index]) ** 2, axis=1
            )

        old_residual = residual
        residual = np.sum(data_distances_to_centroids)

        no_iterations += 1
    return centroids, min_distance


def kernelized_k_means_clustering(data, clusters, kernel=polynomial_kernel):

    """

    Cluster data using kernelized K-means clustering algorithm

    data: Input data as flattened (N x 1 array)
    clusters: No. of clusters (dict); keys represents the label of clusters and values are indices
    kernel: A kernel function of your own. Default is polynomial kernel.

    Output:
    -------
    Clustered data (dict); keys are the cluster labels and values are the pixel index belonging to a cluster


    """
    distance_to_centers = np.zeros((len(data), len(clusters)))

    tol = 1
    previous_tol = 2

    A = kernel(data, data)

    no_of_iterations = 0

    while np.abs(tol) > 0 and np.abs(previous_tol - tol) > 0.0:
        no_of_iterations += 1

        for cluster_label, cluster in clusters.items():
            B = np.array(
                [2 / len(cluster) * sum(kernel(x, data[cluster])) for x in data]
            )
            C = 1 / (len(cluster) ** 2) * sum(kernel(data[cluster], data[cluster]))
            distance_to_centers[:, cluster_label] = A - B + C

        minimum_distance = np.argmin(distance_to_centers, axis=1)

        previous_tol = tol
        tol = sum(
            [
                distance[minimum_distance[i]]
                for i, distance in enumerate(distance_to_centers)
            ]
        )

        for i, dist in enumerate(np.unique(minimum_distance)):
            clusters[dist] = np.where(minimum_distance == dist)[0]

    print("Converged in: %d iterations", no_of_iterations)

    return clusters