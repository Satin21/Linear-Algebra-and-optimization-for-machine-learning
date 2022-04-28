# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 19:52:37 2022

@author: sofiakotti
"""

import time
t1 = time.time()

import pandas as pd
import numpy as np
#import sklearn
import numpy.random as rnd
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
# from scipy import sparse as sp
# import scipy as sci
# from scipy import io
# import math
# from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn import metrics
# from scipy.sparse import csgraph

# Import dataset
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

#%%
t2 = time.time()

# Convert data to arrays
Xnp = pd.DataFrame.to_numpy(X)
ynp = pd.DataFrame.to_numpy(y)

# Choose number of samples to work with
choose_num_samples = 10000
Xnew = Xnp[:choose_num_samples, :]
ynew = ynp[:choose_num_samples]

# Normalize features
Xnew = Xnew / 255;

#%% Function definition
# Kernel function
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

#%%
K = 10 # Number of clusters
[num_samples, num_features] = np.shape(Xnew)

kernel_type = 'euclidean'
param = 1  # kernel function parameter
#%% Initialize centroids
randidx = rnd.permutation(num_samples)[:K]  # randomly
# randidx = np.arange(K)        # take the first K samples as initial centroids
centroids = Xnew[randidx,:] # initial centroids
#%%
max_iter = 50    # Maximum number of iterations

# Initialize
idx = np.zeros(num_samples)
K1 = np.zeros(num_samples)
obj_f = np.zeros(max_iter)
# temp_cost = np.zeros(num_samples)
K3 = np.zeros(K)

#%% First iteration
# K(c_i, c_i) calculations
for j in np.arange(K):
    K3[j] = kernel_f(centroids[j, :], centroids[j, :], kernel_type, param)
    
for idx_s in np.arange(num_samples):
    # K(x_i, x_i) calculations
    K1[idx_s] = kernel_f(Xnew[idx_s, :], Xnew[idx_s, :], kernel_type, param)
    # K(x_i, c_i) calculations
    K2 = kernel_f(Xnew[idx_s, :], centroids.T, kernel_type, param) 
    idx[idx_s] = np.argmin(K1[idx_s]-2 * K2+K3)
    obj_f[0] += np.min(K1[idx_s]-2 * K2+K3)
    # temp_cost[idx_s] = np.min(K1[idx_s]-2 * K2+K3)
    
#%% All following iterations
iter = 1
diff_obj_f = -1
# idx_sample = np.arange(num_samples)

# print(idx)
#%% 
# Initialize
idx_ck = np.zeros((num_samples, K), dtype=bool)
Ck = np.zeros(K, dtype=int)
K3 = np.zeros(K)
#
K_all = np.zeros((num_samples, num_samples), dtype='f4')
#%%
while iter < max_iter and diff_obj_f < 0:
    # Create upper triangular similarity matrix K
    for idx_s in np.arange(num_samples):
        K_all[idx_s, idx_s :] = kernel_f(Xnew[idx_s, :], Xnew[idx_s:, :].T, kernel_type, param)
    # Extend matrix K to symmetric
    for idx_s in np.arange(num_samples):
        K_all[idx_s+1 :, idx_s] = K_all[idx_s, idx_s+1 :] 
    score = np.zeros((num_samples, K)) 
    for j in np.arange(K):
        idx_ck[:, j] = idx == j
        Ck[j] = np.sum(idx_ck[:, j]) 
        in_ck = np.nonzero(idx == j)[0]
        # print(in_ck)
        K3[j] = np.sum(K_all[np.ix_(in_ck, in_ck)])
        # print(K_all[np.ix_(in_ck, in_ck)])
        for idx_s in np.arange(num_samples):  
            K2 = np.sum(K_all[idx_s, in_ck])
            # print(K2)
            score[idx_s, j] = K1[idx_s] - 2 * K2/Ck[j] + K3[j]/(Ck[j]**2) 
            # print(score)
    idx_update = np.argmin(score, axis=1)
    obj_f[iter] = np.sum(np.min(score, axis=1))
    
    diff_obj_f = obj_f[iter] - obj_f[iter - 1]
    iter  += 1
    idx = idx_update

#%%
t3 = time.time()
print('Time elapsed with data loading [s]: ', t3 - t1)
print('Time elapsed without data loading [s]: ', t3 - t2)

#%% Accuracy calculation
A = np.zeros(K)
freq_m = np.zeros(K)
for idx_c in np.arange(K):
    freq_m[idx_c] = mostFrequent(ynew[idx == idx_c])[1]
    A[idx_c] = freq_m[idx_c]
    
Acc = sum(A) / num_samples

#%% Show metrics
print('Accuracy: ', Acc)
print('Completeness: ', metrics.completeness_score(ynew, idx))
print('Homogeneity: ', metrics.homogeneity_score(ynew, idx))