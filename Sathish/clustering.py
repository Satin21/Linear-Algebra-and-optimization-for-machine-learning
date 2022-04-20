import numpy as np

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

    while tol > 0 and (previous_tol - tol) > 0.0:
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

    return clusters