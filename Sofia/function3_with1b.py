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
    centroids = Xcompr[randidx,:] # initial centroids

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
        # K1[idx_s] = kernel_f(Xcompr[idx_s, :], Xcompr[idx_s, :], kernel_type, param)

        # K(x_i, c_i) calculations from the specific sample to all initial centroids
        K2 = kernel_f(Xcompr[idx_s, :], centroids.T, kernel_type, param) 

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
        K_all[idx_s, idx_s :] = kernel_f(Xcompr[idx_s, :], Xcompr[idx_s:, :].T, kernel_type, param)
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