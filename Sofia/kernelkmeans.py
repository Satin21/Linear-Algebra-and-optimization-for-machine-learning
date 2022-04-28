# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 13:22:56 2022

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

from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
#%%
t2 = time.time()
Xnp = pd.DataFrame.to_numpy(X)
ynp = pd.DataFrame.to_numpy(y)
Xnew = Xnp[:10000, :]
ynew = ynp[:10000]
# Xnew = Xnp
# ynew = ynp
Xnew = Xnew / 255;

#%%
# tst = sci.io.loadmat('X.mat')
# Xnew = tst['X'][:200, :]
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
    elif typ == 'euclidean':
        if X.ndim == 1:
            X = np.array([X]).T
            if Y.ndim == 1:
                Y = np.array([Y]).T
        else:
            if Y.ndim == 1:
                Y = np.array([Y]).T
        return X.T @ Y
    
    

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
kernel_type = 'polynomial'
param = 1
#%% Initialize centroids
randidx = np.arange(K) #rnd.permutation(num_samples)[:K]
#%%
centroids = Xnew[randidx,:] #initial_centroids
#%%
max_iter = 20
idx = np.zeros(num_samples)
K1 = np.zeros(num_samples)
obj_f = np.zeros(max_iter)
temp_cost = np.zeros(num_samples)
K3 = np.zeros(K)
for j in np.arange(K):
    K3[j] = kernel_f(centroids[j, :], centroids[j, :], kernel_type, param)
for idx_s in np.arange(num_samples):
    K1[idx_s] = kernel_f(Xnew[idx_s, :], Xnew[idx_s, :], kernel_type, param)
    K2 = kernel_f(Xnew[idx_s, :], centroids.T, kernel_type, param) 
    idx[idx_s] = np.argmin(K1[idx_s]-2 * K2+K3)
    obj_f[0] += np.min(K1[idx_s]-2 * K2+K3)
    temp_cost[idx_s] = np.min(K1[idx_s]-2 * K2+K3)
    
#%%

iter = 1
diff_obj_f = -1
idx_sample = np.arange(num_samples)

# print(idx)
#%%
idx_ck = np.zeros((num_samples, K), dtype=bool)
Ck = np.zeros(K, dtype=int)
K3 = np.zeros(K)
#%%
while iter < max_iter and diff_obj_f < 0:
    for j in np.arange(K):
        idx_ck[:, j] = idx == j
        Ck[j] = np.sum(idx_ck[:, j])    
        in_ck = np.nonzero(idx == j)[0]
        # print(in_ck)
        K3_temp = np.zeros(Ck[j])
        for k in np.arange(Ck[j]):
            K3_temp[k] = np.sum(kernel_f(Xnew[in_ck[k], :], Xnew[idx_ck[:, j], :].T, kernel_type, param))
            # print(kernel_f(Xnew[in_ck[k], :], Xnew[idx_ck[:, j], :].T, kernel_type, param))
        # print(K3_temp)
        # print(K3)
        K3[j] = np.sum(K3_temp)  
# 
    idx_update = np.zeros(num_samples)
    for idx_s in np.arange(num_samples):
        score = np.zeros(K)
        tmp2 = kernel_f(Xnew[idx_s, :], Xnew.T, kernel_type, param)
        for j in np.arange(K):
            K2 = np.sum(tmp2.T[idx_ck[:, j]])
            # print(K2)
            score[j] = K1[idx_s] - 2 * K2/Ck[j] + K3[j]/(Ck[j]**2) 
        # print(score)
        idx_update[idx_s] = np.argmin(score)
        obj_f[iter] += np.min(score)
    
    diff_obj_f = obj_f[iter] - obj_f[iter - 1]
    iter  += 1
    idx = idx_update
    # print(idx)
#%%
# for idx_c in np.arange(K):
#     Ximage = np.resize(centroids[idx_c, :], (28, 28))
#     imgplot = plt.imshow(Ximage)
#     plt.show()
#%%
t3 = time.time()
print(t3 - t1)
print(t2 - t1)
#%%
a = 0
for idx_c in np.arange(K): 
    a += sum(idx == idx_c)
#%% Accuracy calculation
m = np.zeros(K)
A = np.zeros(K)
freq_m = np.zeros(K)
for idx_c in np.arange(K):
    m[idx_c], freq_m[idx_c] = mostFrequent(ynew[idx == idx_c])
    A[idx_c] = freq_m[idx_c]
    
Acc = sum(A) / num_samples
print(Acc)
    
#%%
# t = time.time()
# for j in np.arange(1):
#     idx_ck[:, j] = idx == j
#     Ck[j] = np.sum(idx_ck[:, j])  
#     in_ck = np.nonzero(idx == j)[0]
#     for k in np.arange(Ck[j]):
#         for l in np.arange(Ck[j]):
#             K3[j] += kernel_f(Xnew[in_ck[k], :], Xnew[in_ck[l], :], kernel_type, param)
#         # print(k)
# elapsed = time.time() - t
# print(elapsed)

#%%
# t = time.time()
# for j in np.arange(1):
#     idx_ck[:, j] = idx == j
#     Ck[j] = np.sum(idx_ck[:, j]) 
#     in_ck = np.nonzero(idx == j)[0]
#     K3_temp = np.zeros(Ck[j])
#     for k in np.arange(Ck[j]):
#         K3_temp[k] = np.sum(kernel_f(Xnew[in_ck[k], :], Xnew[idx_ck[:, j], :].T, kernel_type, param))
#     # print(K3)
#     K3[j] = np.sum(K3_temp)
# elapsed = time.time() - t
# print(elapsed)

#%%
# idx_s = 0
# K21 = np.zeros(K)
# K22 = np.zeros(K)
# for j in np.arange(K):
#     idx_ck[:, j] = idx == j
    
# t = time.time()    
# for j in np.arange(K):
#     tmp = kernel_f(Xnew[idx_s, :], Xnew[idx_ck[:, j], :].T, kernel_type, param)
#     # print(K2)
#     K21[j] = np.sum(tmp)
# print(time.time() - t)   

# t = time.time()  
# tmp2 = kernel_f(Xnew[idx_s, :], Xnew.T, kernel_type, param)
# for j in np.arange(K):
#     K22[j] = np.sum(tmp2.T[idx_ck[:, j]])
# print(time.time() - t)  

#%%
    
    #%%
    # while iter < max_iter and diff_obj_f < 0:
    # for j in np.arange(K):
    #     idx_ck[:, j] = idx == j
    #     Ck[j] = np.sum(idx_ck[:, j])    
    #     K3_temp = np.zeros(Ck[j])
    #     for k in np.arange(Ck[j]):
    #         K3_temp[k] = np.sum(kernel_f(Xnew[np.where(idx_ck[:, j])[0][k], :], Xnew[idx_ck[:, j], :].T, kernel_type, param))
    #     # print(K3)
    #     K3[j] = np.sum(K3_temp)       
    # idx_update = np.zeros(num_samples)
    # for idx_s in np.arange(1):
    #     score = np.zeros(K)
    #     for j in np.arange(K):
    #         K2 = kernel_f(Xnew[idx_s, :], Xnew[idx_ck[:, j], :].T, kernel_type, param)
    #         # print(K2)
    #         K2 = np.sum(K2)
    #         # print(K2)
    #         score[j] = K1[idx_s] - 2 * K2/Ck[j] + K3[j]/(Ck[j]**2) 
    #     # print(score)
    #     idx_update[idx_s] = np.argmin(score)
    #     obj_f[iter] += np.min(score)
    
    # diff_obj_f = obj_f[iter] - obj_f[iter - 1]
    # iter  += 1
    # idx = idx_update