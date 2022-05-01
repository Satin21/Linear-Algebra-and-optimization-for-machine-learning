def optimize_L_kernel(idx_run, choose_samples, l, ytrue, kernel_type = 'polynomial', param = 3):
    
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
    
    #%% Choose spectral clustering options
    version = 'normalized'
    #%% Compress the images
    Xcompr = np.zeros(np.shape(Xnew))
    for idx_s in np.arange(num_samples):
        data = np.resize(Xnew[idx_s, :], (28, 28))
        Xcompr[idx_s, :] = np.resize(SVD(data, L=l, return_compressed = True), (1, 784)) 
    
    #%% Initialize similarity matrix including all kernel distances
    K_all = np.zeros((num_samples, num_samples), dtype='f4')
    
    #%% Fill in the upper triangular part of the similarity matrix
    for idx_s in np.arange(num_samples):
        K_all[idx_s, idx_s :] = kernel_f(Xcompr[idx_s, :], Xcompr[idx_s:, :].T, kernel_type, param)
    # Extend K to symmetric
    for idx_s in np.arange(num_samples):
        K_all[idx_s+1 :, idx_s] = K_all[idx_s, idx_s+1 :] 
        
    D = np.diag(np.sum(K_all, axis=0))
    L = D - K_all
    
    if version == 'normalized':
        Dneg_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        L = Dneg_sqrt @ L @ Dneg_sqrt
    
    # Find the eigenvalues and eigenvectors
    eigenValues, eigenVectors = eigh(L) 
    # Sort the eigenvalues in increasing order and the eigenvectors respectively
    idx_eig = eigenValues.argsort()  
    eigenValues = eigenValues[idx_eig]
    eigenVectors = eigenVectors[:,idx_eig]
    # Create H matrix using the eigenvectors corresponding to the lowest K eigenvalues
    H = eigenVectors[:, :K]
    
    #%% Choose k-means options
    [num_rows, num_features] = np.shape(H)
    max_iter = 50
    iter = 0
    diff_obj_f = -1
    obj_f = np.zeros(max_iter)
    
    #%% Initialize centroids
    randidx = rnd.permutation(num_rows)[:K] # randomly
    # randidx = np.arange(K) # take the first K samples as initial centroids
    centroids = H[randidx,:] # initial centroids
    
    #%% k-means iterations for the rows of H matrix
    while iter < max_iter and diff_obj_f < 0:
        # Assign rows to closest centroids
        idx = assign_to_cluster(H, centroids)
        # Update the centroids
        centroids = update_centroids(H, idx, K)
    
        temp = np.linalg.norm(H - centroids[idx, :], ord=2, axis=1)
        # Calculate the value of the objective function at this iteration
        obj_f[iter] = sum(temp**2)
        
        if iter != 0:
            diff_obj_f = obj_f[iter] - obj_f[iter - 1]
        iter  += 1
            
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