import numpy as np


def polynomial_kernel(x1, x2):
    return (x1.T * x2)**2

def K_means_clustering(data, clusters):
    '''
    Cluster data using K-means clustering algorithm

    Input:
    -------
    data: Input data as flattened (N x 1 array)
    clusters: No. of clusters (dict); keys represents the label of clusters and values are empty

    Output:
    -------
    Clustered data (dict); keys are the cluster labels and values are the pixel index belonging to a cluster
    '''
    _, counts = np.unique(data, return_counts = True)
    sorted_counts = np.argsort(counts)[::-1]

    centers = []
    for i in list(clusters.keys()):
        centers.append(np.unique(data)[sorted_counts[i]])

    tol = 1
    previous_tol = 2

    no_of_iterations = 0

    while np.abs(tol) > 0 and np.abs(previous_tol - tol) > 0.0:

        no_of_iterations += 1
        
        distance_to_centers = np.zeros((len(data), len(centers)))

        for i, center in enumerate(centers):
            distance_to_centers[:, i] = (data - center)**2 
        
        clusters = {}
        minimum_distance = np.argmin(distance_to_centers, axis = 1)

        for i, dist in enumerate(np.unique(minimum_distance)):
            clusters[dist] = np.where(minimum_distance == dist)[0]

        assert sum([len(x) for x in clusters.values()]) == len(data)

        previous_tol = tol

        tol = 0.0
        centers = []
        for i in list(clusters.keys()):
            centers.append(sum(data[clusters[i]])/len(clusters[i]))
            tol += sum((data[clusters[i]] - centers[i])**2)

    print("Converged in: %d iterations", no_of_iterations)

    return clusters

def kernelized_k_means_clustering(data, clusters, kernel = polynomial_kernel):

    '''
    
    Cluster data using kernelized K-means clustering algorithm

    data: Input data as flattened (N x 1 array)
    clusters: No. of clusters (dict); keys represents the label of clusters and values are indices
    kernel: A kernel function of your own. Default is polynomial kernel.

    Output:
    -------
    Clustered data (dict); keys are the cluster labels and values are the pixel index belonging to a cluster

    
    '''
    distance_to_centers = np.zeros((len(data), len(clusters)))

    tol = 1
    previous_tol = 2

    A = kernel(data, data)

    no_of_iterations = 0

    while np.abs(tol) > 0 and np.abs(previous_tol - tol) > 0.0:
        no_of_iterations += 1

        for cluster_label, cluster in clusters.items():
            B = np.array([2/len(cluster) * sum(kernel(x, data[cluster])) for x in data])
            C = 1/(len(cluster)**2) * sum(kernel(data[cluster], data[cluster]))
            distance_to_centers[:, cluster_label] = A - B + C

        minimum_distance = np.argmin(distance_to_centers, axis = 1)

        previous_tol = tol
        tol = sum([distance[minimum_distance[i]] for i, distance in enumerate(distance_to_centers)])

        for i, dist in enumerate(np.unique(minimum_distance)):
            clusters[dist] = np.where(minimum_distance == dist)[0]
        
          
    print("Converged in: %d iterations", no_of_iterations)
    
    return clusters