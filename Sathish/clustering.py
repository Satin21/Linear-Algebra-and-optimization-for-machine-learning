import numpy as np
from itertools import combinations
import random


def polynomial_kernel(x1, x2):
    return (x1.T * x2) ** 2


def K_means_clustering(X, y, subset, max_iter=50, tolerance=1e-16):

    max_distance = []
    positions = []
    for i in range(100):
        random_positions = random.sample(range(len(X)), 100)
        combos = []
        for i in range(10):
            combos.append(
                sum(
                    [
                        np.sum((X[indices[0]] - X[indices[1]]) ** 2)
                        for indices in list(
                            combinations(random_positions[10 * i : 10 * (i + 1)], 2)
                        )
                    ]
                )
            )
            max_index = np.argmax(combos)
            positions.append(random_positions[10 * max_index : 10 * (max_index + 1)])
            max_distance.append(combos[np.argmax(combos)])

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
    residual = old_residual + 1
    no_iterations = 0

    while np.abs(old_residual - residual) > tolerance and (no_iterations < max_iter):

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
