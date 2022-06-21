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
        for i in range(nclusters):
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

def optimize_L(l, ytrue):
    
    with open('/home/srangaswamykup/LAOML/data.pkl', 'rb') as outfile:
        data = pickle.load(outfile)
    
    compressed = np.array(list(map(lambda x : SVD(x.reshape(28, -1), L = l).flatten(), data)))
    
    t1 = time.time()
    centroids, min_distance = K_means_clustering(compressed, 10, subset = len(compressed), tolerance = 1e7)
    t2 = time.time()
    return find_accuracy(10, ytrue, min_distance), t2-t1


def optimize_L_kernel(idx_run, choose_samples, l, ytrue, kernel_type = 'polynomial', param = 2):
    
    with open('/home/srangaswamykup/LAOML/data.pkl', 'rb') as outfile:
        data = pickle.load(outfile)
    
    Xnp = data
    ynp = ytrue
    
    K = 10
    max_iter = 50
    
    # Use the corresponding dataset
    Xnew = Xnp[choose_samples[idx_run, :], :]
    ynew = ynp[choose_samples[idx_run, :]]

    # Normalize features to [0,1]
    Xnew = Xnew / 255;
    [num_samples, num_features] = np.shape(Xnew)

    t1 = time.time()

    #%% Compress the images
    Xcompr = np.zeros(np.shape(Xnew))
    for idx_s in np.arange(num_samples):
        data = np.resize(Xnew[idx_s, :], (28, 28))
        Xcompr[idx_s, :] = np.resize(SVD(data, L=l, return_compressed = True), (1, 784)) 

    #%% Initialize centroids
    randidx = rnd.permutation(num_samples)[:K]  # randomly
    # randidx = np.arange(K)        # take the first K samples as initial centroids
    centroids = Xnew[randidx,:] # initial centroids

    #%% Initialize
    idx = np.zeros(num_samples)
    K1 = np.zeros(num_samples)
    obj_f = np.zeros(max_iter)
    K3 = np.zeros(K)

    #%% First iteration         
    for j in np.arange(K):
        # K(c_i, c_i) calculations
        K3[j] = kernel_f(centroids[j, :], centroids[j, :], kernel_type, param)

    for idx_s in np.arange(num_samples):
        # # K(x_i, x_i) calculations can be actually skipped to speed up the algorithm
        # # because they don't affect the minimization problem
        # K1[idx_s] = kernel_f(Xnew[idx_s, :], Xnew[idx_s, :], kernel_type, param)

        # K(x_i, c_i) calculations from the specific sample to all initial centroids
        K2 = kernel_f(Xnew[idx_s, :], centroids.T, kernel_type, param) 

        # Choosing the closest centroid per sample
        idx[idx_s] = np.argmin(-2 * K2 + K3)
        # Calculating the cost of the assignment 
        obj_f[0] += np.min(-2 * K2 + K3)

        # # In case K(x_i, x_i) calculations were included to find the value of the 
        # # original objective function
        # idx[idx_s] = np.argmin(K1[idx_s]-2 * K2+K3)
        # obj_f[0] += np.min(K1[idx_s]-2 * K2+K3)

    #%% All following iterations
    idx_iter = 1
    diff_obj_f = -1

    #%% 
    # Initialize
    idx_ck = np.zeros((num_samples, K), dtype=bool) # which samples belong 
    # currently to which cluster (binary matrix)
    Ck = np.zeros(K, dtype=int)     # no. of samples per cluster
    K3 = np.zeros(K)
    #
    # Similarity matrix including all kernel distances
    K_all = np.zeros((num_samples, num_samples), dtype='f4')

    #%% Fill the similarity matrix
    for idx_s in np.arange(num_samples):
        K_all[idx_s, idx_s :] = kernel_f(Xnew[idx_s, :], Xnew[idx_s:, :].T, kernel_type, param)
        # Extend matrix K to symmetric
    for idx_s in np.arange(num_samples):
        K_all[idx_s+1 :, idx_s] = K_all[idx_s, idx_s+1 :]

    # Kernelized k-means iterations
    while idx_iter < max_iter and diff_obj_f < 0:
        # Create upper triangular similarity matrix K  
        score = np.zeros((num_samples, K)) # Score per sample per cluster
        for j in np.arange(K):
            idx_ck[:, j] = idx == j
            Ck[j] = np.sum(idx_ck[:, j]) # no. of samples per cluster
            in_ck = np.nonzero(idx == j)[0] # get samples indices for all samples in the cluster
            K3[j] = np.sum(K_all[np.ix_(in_ck, in_ck)]) # Sum of kernelized pairwise 
            # distances for all samples in the cluster 

            for idx_s in np.arange(num_samples):  
                K2 = np.sum(K_all[idx_s, in_ck]) # Sum of kernelized distances 
                # from the sample to all samples in the cluster 
                score[idx_s, j] = - 2 * K2/Ck[j] + K3[j]/(Ck[j]**2) 

                # # In case K(x_i, x_i) calculations were included to find the value of the 
                # # original objective function
                # score[idx_s, j] = K1[idx_s] - 2 * K2/Ck[j] + K3[j]/(Ck[j]**2) 

        # Choosing the closest centroid per sample
        idx = np.argmin(score, axis=1)
        # Calculating the cost of the assignment 
        obj_f[idx_iter] = np.sum(np.min(score, axis=1))
        # Difference in objective function
        diff_obj_f = obj_f[idx_iter] - obj_f[idx_iter - 1]
        idx_iter  += 1

    #%%
    t2 = time.time()
    
    #%% Accuracy calculation
    # A = np.zeros(K)
    # freq_m = np.zeros(K)
    # for idx_c in np.arange(K):
    #     freq_m[idx_c] = mostFrequent(ynew[idx == idx_c])[1]
    #     A[idx_c] = freq_m[idx_c]

    # Acc = sum(A) / num_samples
    
    return find_accuracy(10, ynew, idx), t2-t1