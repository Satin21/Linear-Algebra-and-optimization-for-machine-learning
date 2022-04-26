# @jit(nopython=True, parallel=True)
def kMeans(X, y_true, K: int, metric: str):
    log('Starting K-means clustering using the metric \'%s\'' % metric)
    
    # Randomly initialize the centroids c
    N, M = X.shape
    X = DataFrame(data=X)
    c = X.sample(K).reset_index(drop=True)
    y = [-1] * N  # Labels

    ePrev = M * N + 1
    eNew = ePrev - 1
    while ePrev - eNew > 0:
        ePrev = eNew
        eNew = 0

        # Assign x_i to cluster k = argmin_j || x_i - c_j ||
        for i, x in X.iterrows():
            if metric == 'gauss':
                gamma = .05
                dists = cdist([x.array], c, 'sqeuclidean')
                dists = exp(-gamma * dists)
            else:
                dists = cdist([x.array], c, metric)
            y[i] = argmin(dists)

        # Update the centroid positions
        for k in range(K):
            Ck = X.loc[[i for i, yi in enumerate(y) if yi == k]]  # Compute Ck = {x_i where the label y_i = k}
            if len(Ck) > 0:
                c.loc[k] = Ck.mean()                            # Centroid = the mean of the elements in Ck
                eNew += cdist([c.loc[k]], Ck).sum()             # Compute the error
            else:
                # Try to find a good position for this centroid
                c.loc[k] = c.mean()
    
    # Determine the accuracy
    a = 0
    for k in range(K):
        Lk = y_true.loc[[i for i, yi in enumerate(y) if yi == k]]  # Lk = {y_true_i : y_i = k}
        Lk = DataFrame(data=Lk)
        cnt = Lk.value_counts()
        cnt = cnt.values
        if len(cnt) > 0:
            a += cnt[0]
    a /= N
    
#     showCentroids(c)

    return y, a  # Return the labels
