# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 16:42:58 2022

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
from scipy import sparse as sp
import scipy as sci
from scipy import io
import math
from scipy.spatial.distance import euclidean, pdist, squareform
from numpy.linalg import eig
from numpy.linalg import eigh
from numpy.linalg import inv
from scipy.linalg import sqrtm 
from sklearn import metrics
from scipy.sparse import csgraph

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#%%
t2 = time.time()

# Convert data to arrays
Xnp = pd.DataFrame.to_numpy(X)
ynp = pd.DataFrame.to_numpy(y)

# Choose number of samples to work with
choose_num_samples = 5000
Xnew = Xnp[:choose_num_samples, :]
ynew = ynp[:choose_num_samples]

# Normalize features
Xnew = Xnew / 255;

#%% Function definition
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

def update_centroids(X, idx, K):
    num_features = np.shape(X)[1]
    C = np.zeros((K,num_features))
    for idx_c in np.arange(K):
        C[idx_c, :] = np.mean(X[np.where(idx == idx_c)[0], :], axis = 0)
    return C

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
#%% 
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
#Xnew = rnd.rand(4,8)
[num_samples, num_features] = np.shape(Xnew)

K = 10
kernel_type = 'gaussian'
param = 1/50
#%%
K_all = np.zeros((num_samples, num_samples), dtype='f4')
#%%
t = time.time()
# Create upper triangular K
for idx_s in np.arange(num_samples):
    K_all[idx_s, idx_s :] = kernel_f(Xnew[idx_s, :], Xnew[idx_s:, :].T, kernel_type, param)
# Extend K to symmetric
for idx_s in np.arange(num_samples):
    K_all[idx_s+1 :, idx_s] = K_all[idx_s, idx_s+1 :] 
print(time.time() - t)

#%%
# K_all_save = K_all
# K_all = K_all_save
K_all[K_all < 0.2] = 0
K_all[K_all >= 0.2] = 1
D = np.diag(np.sum(K_all, axis=0))
L = D - K_all
Dneg_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
L = Dneg_sqrt @ L @ Dneg_sqrt

#
eigenValues, eigenVectors = eigh(L)
idx_eig = eigenValues.argsort()  
eigenValues = eigenValues[idx_eig]
eigenVectors = eigenVectors[:,idx_eig]
H = eigenVectors[:, :K]

#%%
[num_rows, num_features] = np.shape(H)
max_iter = 50
iter = 0
diff_obj_f = -1
# idx_row = np.arange(num_rows)
obj_f = np.zeros(max_iter)
#%% Initialize centroids
# randidx = np.arange(start=14, stop=24)
randidx = rnd.permutation(num_samples)[:K]
# randidx = np.arange(K)
#%%
centroids = H[randidx,:] #initial_centroids
#%%
while iter < max_iter and diff_obj_f < 0:
    idx = assign_to_cluster(H, centroids)
    # print(idx)
    centroids = update_centroids(H, idx, K)

    temp = np.linalg.norm(H - centroids[idx, :], ord=2, axis=1)
    obj_f[iter] = sum(temp**2)
    if iter != 0:
        diff_obj_f = obj_f[iter] - obj_f[iter - 1]
    iter  += 1
    # print(idx)

#%%
t3 = time.time()
print('Time elapsed with data loading [s]: ', t3 - t1)
print('Time elapsed without data loading [s]: ', t3 - t2)

#%% Accuracy calculation
m = np.zeros(K)
A = np.zeros(K)
freq_m = np.zeros(K)
for idx_c in np.arange(K):
    m[idx_c], freq_m[idx_c] = mostFrequent(ynew[idx == idx_c])
    A[idx_c] = freq_m[idx_c]
    
Acc = sum(A) / num_samples

#%% # Show metrics
print('Accuracy: ', Acc)
print('Completeness: ', metrics.completeness_score(ynew, idx))
print('Homogeneity: ', metrics.homogeneity_score(ynew, idx))