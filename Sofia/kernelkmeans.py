# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:22:56 2022

@author: sofiakotti
"""

import time
t = time.time()

import pandas as pd
import numpy as np
#import sklearn
import numpy.random as rnd
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from scipy import sparse as sp
import scipy as sci
import math

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#%%
Xnp = pd.DataFrame.to_numpy(X)
ynp = pd.DataFrame.to_numpy(y)
# Xnew = Xnp[:700, :]
Xnew = Xnp
Xnew = Xnew / 255;

#%%
# tst = sci.io.loadmat('X.mat')
# Xnew = tst['X'][:700, :]
#%%
def assign_to_cluster(X, C):
    num_features = np.shape(X)[1]
    #print(num_features)
    K = np.shape(C)[0]
    #print(K)
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
       # print(C)
    a = X[np.where(idx == idx_c)[0], :]
    row, col = np.shape(a)
    b = np.reshape(np.kron(a, np.ones((1, col))), (row, col, col), order="F")
    repa = a[:,:,np.newaxis]
    repa = np.tile(repa, (1, 1, np.shape(a)[1]))
    return C
#%%
def kernel_f(X, Y, typ, param): # WILL NOT WORK IF CALLED WITH SCALAR ARGUMENTS
    if  typ == 'polynomial':
        if  X.ndim == 1:
            X = np.array([X]).T
        if  Y.ndim == 1:
            Y = np.array([Y]).T
        return (1+X.T @ Y)**param
    elif typ == 'gaussian':
        if X.ndim == 1:
            X = np.array([X]).T
            if Y.ndim == 1:
                Y = np.array([Y]).T
            diff = Y - X
            nrm = np.linalg.norm(diff, ord=2, axis=0)
        else:
            if Y.ndim == 1:
                Y = np.array([Y]).T
            row1, col1 = np.shape(X)
            row2, col2 = np.shape(Y)
            b = np.reshape(np.kron(X, np.ones((1, col2))), (row1, col2, col1), order="F")
            repY = Y[:,:,np.newaxis]
            repY = np.tile(repY, (1, 1, col1))
            diff = repY - b
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

# Ximage = np.resize(Xnew[3000:], (28, 28))
# imgplot = plt.imshow(Ximage)
# plt.show()

K = 10
#%% Initialize centroids
# randidx = np.arange(start=14, stop=24)
randidx = rnd.permutation(num_samples)[:K]
#%%
centroids = Xnew[randidx,:] #initial_centroids


#%%
max_iter = 50
iter = 0
diff_obj_f = -1
idx_sample = np.arange(num_samples)
obj_f = np.zeros(max_iter)
while iter < max_iter and diff_obj_f < 0:
    idx = assign_to_cluster(Xnew, centroids)
   
    #print(idx)
    centroids = update_centroids(Xnew, idx, K)

    temp = np.linalg.norm(Xnew - centroids[idx, :], ord=2, axis=1)
    obj_f[iter] = sum(temp**2)
    if iter != 0:
        diff_obj_f = obj_f[iter] - obj_f[iter - 1]
    iter  += 1
    
#%%
for idx_c in np.arange(K):
    Ximage = np.resize(centroids[idx_c, :], (28, 28))
    imgplot = plt.imshow(Ximage)
    plt.show()
#%%
elapsed = time.time() - t
#%%
a = 0
for idx_c in np.arange(K): 
    a += sum(idx == idx_c)
#%% Accuracy calculation
m = np.zeros(K)
A = np.zeros(K)
freq_m = np.zeros(K)
for idx_c in np.arange(K):
    m[idx_c], freq_m[idx_c] = mostFrequent(ynp[idx == idx_c])
    A[idx_c] = freq_m[idx_c]
    
Acc = sum(A) / num_samples
print(Acc)
    