# -*- coding: utf-8 -*-
"""
Script that implements that kernelized k-means algorithm on the MNIST dataset,
as described on page 128 of the notes. All necessary functions for the script 
are defined below.
"""

#%% Library import
import pandas as pd
import numpy as np
import numpy.random as rnd
from sklearn import metrics
import time

#%% Dataset import
from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

#%% Function definition

# Kernel functions
def kernel_f(X, Y, typ, param):  # Assumes that samples have >1 features, so will
    # not work if called with scalar arguments
    if typ == "polynomial":
        if X.ndim == 1:
            X = np.array([X]).T
        if Y.ndim == 1:
            Y = np.array([Y]).T
        return (1 + X.T @ Y) ** param
    elif typ == "euclidean":
        if X.ndim == 1:
            X = np.array([X]).T
        if Y.ndim == 1:
            Y = np.array([Y]).T
        return X.T @ Y
    elif typ == "gaussian":
        if X.ndim == 1 and Y.ndim == 1:
            pass
        elif X.ndim == 1:
            X = np.array([X]).T
        diff = Y - X
        nrm = np.linalg.norm(diff, ord=2, axis=0)
        return np.exp(-param * nrm**2)


# Function to find the most frequent number per cluster
# source: https://www.geeksforgeeks.org/frequent-element-array/
def mostFrequent(arr):
    n = len(arr)
    # Insert all elements in Hash.
    Hash = dict()
    for i in range(n):
        if arr[i] in Hash.keys():
            Hash[arr[i]] += 1
        else:
            Hash[arr[i]] = 1

    # find the max frequency
    max_count = 0
    res = -1
    for i in Hash:
        if max_count < Hash[i]:
            res = i
            max_count = Hash[i]

    return res, max_count


#%% Start-up
# Convert data to arrays
Xnp = pd.DataFrame.to_numpy(X)
ynp = pd.DataFrame.to_numpy(y)

# Choose number of samples to work with
total_num_samples = Xnp.shape[0]
choose_num_samples = 10000
choose_samples = rnd.permutation(total_num_samples)[
    :choose_num_samples
]  # use random samples of the dataset
# choose_samples = np.arange(choose_num_samples) # use the first samples of the dataset
Xnew = Xnp[choose_samples, :]
ynew = ynp[choose_samples]

# Normalize features to [0,1]
Xnew = Xnew / 255

t1 = time.time()
#%% Choose algorithm options
K = 10  # Number of clusters
[num_samples, num_features] = np.shape(Xnew)

kernel_type = "polynomial"  # or 'gaussian' or 'euclidean'
param = 2  # kernel function parameter

max_iter = 50  # Maximum number of iterations

#%% Initialize centroids
randidx = rnd.permutation(num_samples)[:K]  # randomly
# randidx = np.arange(K)        # take the first K samples as initial centroids
centroids = Xnew[randidx, :]  # initial centroids

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
idx_ck = np.zeros((num_samples, K), dtype=bool)  # which samples belong
# currently to which cluster (binary matrix)
Ck = np.zeros(K, dtype=int)  # no. of samples per cluster
K3 = np.zeros(K)
#
# Initialize similarity matrix including all kernel distances
K_all = np.zeros((num_samples, num_samples), dtype="f4")

#%% Fill in the upper triangular part of the similarity matrix
for idx_s in np.arange(num_samples):
    K_all[idx_s, idx_s:] = kernel_f(
        Xnew[idx_s, :], Xnew[idx_s:, :].T, kernel_type, param
    )
# Extend matrix K to symmetric
for idx_s in np.arange(num_samples):
    K_all[idx_s + 1 :, idx_s] = K_all[idx_s, idx_s + 1 :]

# Kernelized k-means iterations
while idx_iter < max_iter and diff_obj_f < 0:
    # Create upper triangular similarity matrix K
    score = np.zeros((num_samples, K))  # Score per sample per cluster
    for j in np.arange(K):
        idx_ck[:, j] = idx == j
        Ck[j] = np.sum(idx_ck[:, j])  # no. of samples per cluster
        in_ck = np.nonzero(idx == j)[
            0
        ]  # get samples indices for all samples in the cluster
        K3[j] = np.sum(K_all[np.ix_(in_ck, in_ck)])  # Sum of kernelized pairwise
        # distances for all samples in the cluster

        for idx_s in np.arange(num_samples):
            K2 = np.sum(K_all[idx_s, in_ck])  # Sum of kernelized distances
            # from the sample to all samples in the cluster
            score[idx_s, j] = -2 * K2 / Ck[j] + K3[j] / (Ck[j] ** 2)

            # # In case K(x_i, x_i) calculations were included to find the value of the
            # # original objective function
            # score[idx_s, j] = K1[idx_s] - 2 * K2/Ck[j] + K3[j]/(Ck[j]**2)

    # Choosing the closest centroid per sample
    idx = np.argmin(score, axis=1)
    # Calculating the cost of the assignment
    obj_f[idx_iter] = np.sum(np.min(score, axis=1))
    # Difference in objective function
    diff_obj_f = obj_f[idx_iter] - obj_f[idx_iter - 1]
    idx_iter += 1

#%%
t2 = time.time()
print("Time elapsed without data loading [s]: ", t2 - t1)

#%% Accuracy calculation
A = np.zeros(K)
freq_m = np.zeros(K)
for idx_c in np.arange(K):
    freq_m[idx_c] = mostFrequent(ynew[idx == idx_c])[1]
    A[idx_c] = freq_m[idx_c]

Acc = sum(A) / num_samples

#%% Show metrics
print("Accuracy: ", Acc)
# print('Completeness: ', metrics.completeness_score(ynew, idx))
# print('Homogeneity: ', metrics.homogeneity_score(ynew, idx))
