# -*- coding: utf-8 -*-
"""
Script that implements that spectral clustering algorithm (unnormalized and 
normalized) on the MNIST dataset, as described on page 132-133 of the notes. 
All necessary functions for the script are defined below.
"""

#%% Library import 
import pandas as pd
import numpy as np
#import sklearn
import numpy.random as rnd
from numpy.linalg import eigh
# from numpy.linalg import qr 
from sklearn import metrics
# import numpy.random as rnd
import copy
import time

#%% Dataset import
from sklearn.datasets import fetch_openml
Xall, yall = fetch_openml('mnist_784', version=1, return_X_y=True)

#%% Function definition
# Function necessary for k-means
def assign_to_cluster(X, C):
    num_features = np.shape(X)[1]
    K = np.shape(C)[0]
    repX = X[:,:,np.newaxis]
    repX = np.tile(repX, (1, 1, K)) # Repeat samples K times
    reshC = np.reshape(np.transpose(C), (1, num_features, K))
    diff = repX - reshC;
    dist = np.linalg.norm(diff, ord=2, axis=1)
    idx = np.argmin(dist, axis=1);
    return idx

# Function necessary for k-means
def update_centroids(X, idx, K):
    num_features = np.shape(X)[1]
    C = np.zeros((K,num_features))
    for idx_c in np.arange(K):
        C[idx_c, :] = np.mean(X[np.where(idx == idx_c)[0], :], axis = 0)
    return C

# Kernel functions
def kernel_f(X, Y, typ, param): # WILL NOT WORK IF CALLED WITH SCALAR ARGUMENTS
    if  typ == 'polynomial':
        if  X.ndim == 1:
            X = np.array([X]).T
        if  Y.ndim == 1:
            Y = np.array([Y]).T
        return (1+X.T @ Y)**param
    elif typ == 'euclidean':
        if X.ndim == 1:
            X = np.array([X]).T
        if Y.ndim == 1:
            Y = np.array([Y]).T
        return X.T @ Y
    elif typ == 'gaussian':
        if X.ndim == 1 and Y.ndim == 1:
            pass
        elif X.ndim == 1:
            X = np.array([X]).T
        diff = Y - X
        nrm = np.linalg.norm(diff, ord=2, axis=0)
        return np.exp(- param * nrm**2)
    
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
        if (max_count < Hash[i]):
            res = i
            max_count = Hash[i]
         
    return res, max_count

#%% #%% Start-up
# Convert data to arrays
Xnp = pd.DataFrame.to_numpy(Xall)
ynp = pd.DataFrame.to_numpy(yall)

# Choose number of samples to work with
total_num_samples = Xnp.shape[0]
choose_num_samples = 10000
choose_samples = rnd.permutation(total_num_samples)[:choose_num_samples]  # use random samples of the dataset
# choose_samples = np.arange(choose_num_samples) # use the first samples of the dataset 
Xnew = Xnp[choose_samples, :]
ynew = ynp[choose_samples]

# Normalize features to [0,1]
Xnew = Xnew / 255;

t1 = time.time()

#%% Choose spectral clustering options
[num_samples, num_features] = np.shape(Xnew)

K = 10
kernel_type = 'polynomial' # or 'gaussian' or 'euclidean'
param = 3
version = 'normalized'

#%% Initialize similarity matrix including all kernel distances
K_all = np.zeros((num_samples, num_samples), dtype='f4')

#%% Fill in the upper triangular part of the similarity matrix
for idx_s in np.arange(num_samples):
    K_all[idx_s, idx_s :] = kernel_f(Xnew[idx_s, :], Xnew[idx_s:, :].T, kernel_type, param)
# Extend K to symmetric
for idx_s in np.arange(num_samples):
    K_all[idx_s+1 :, idx_s] = K_all[idx_s, idx_s+1 :] 

# K_all[K_all < lmt] = 0
# K_all[K_all >= lmt] = 1
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
randidx = rnd.permutation(num_samples)[:K] # randomly
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
print('Time elapsed without data loading [s]: ', t2 - t1)

#%% Accuracy calculation
A = np.zeros(K)
freq_m = np.zeros(K)
for idx_c in np.arange(K):
    freq_m[idx_c] = mostFrequent(ynew[idx == idx_c])[1]
    A[idx_c] = freq_m[idx_c]
    
Acc = sum(A) / num_samples

#%% # Show metrics
print('Accuracy: ', Acc)
# print('Completeness: ', metrics.completeness_score(ynew, idx))
# print('Homogeneity: ', metrics.homogeneity_score(ynew, idx))
