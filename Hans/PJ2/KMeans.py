import numpy as np
from pandas import DataFrame
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self):
        self.cluster_centers = []
        self.cluster_y_pred = []

    def __call__(self, X: np.ndarray, y_true: list, K: int):
        # log('Starting K-means clustering')

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

        # Store the cluster centroids
        self.cluster_centers = c

        # Determine the accuracy
        a = 0
        self.cluster_y_pred = [0] * K
        for k in range(K):
            Lk = y_true[[i for i, yi in enumerate(y) if yi == k]]  # Lk = {y_true_i : y_i = k}
            Lk = DataFrame(data=Lk)
            cnt = Lk.value_counts()
            cnt_vals = cnt.values
            if len(cnt_vals) > 0:
                a += cnt_vals[0]

            # Predict the cluster label
            print(k)
            self.cluster_y_pred[k] = cnt.keys()[0][0]
        a /= N

        return y, a  # Return the labels & accuracy

    # Predict the outcomes (0, 1, ...) based on the cluster centers that have been found
    # Each row of X contains 1 sample
    def predict(self, X: np.ndarray):
        y_pred = np.empty((len(X), 1))
        for i, x in enumerate(X):
            # Compute the distance to each cluster center and determine the nearest cluster center
            y_pred[i] = self.cluster_y_pred[np.argmin(np.linalg.norm(x - self.cluster_centers, axis=1))]
        return y_pred
