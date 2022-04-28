# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 18:14:06 2022

@author: sofiakotti
"""

import pandas as pd
import numpy as np
import numpy.random as rnd
from numpy.linalg import qr
from numpy.linalg import eig
import copy

def qr_fact(X):
    m,n = np.shape(X)
    Q = np.zeros((m,n))
    R = np.zeros((n,n))
    R[0,0] = np.linalg.norm(X[:, 0])
    q1 = X[:, 0] / R[0,0]
    Q[:, 0] = q1
    for i in np.arange(1, n):
        q_i = copy.deepcopy(X[:, i]) 
        # print(q_i)
        for j in np.arange(i):
            R[j,i] = np.sum(X[:, i] * Q[:, j])
            q_i -= R[j,i] * Q[:, j]
        R[i,i] = np.linalg.norm(q_i)
        Q[:, i] = q_i / R[i,i]
    return Q, R

def qr_eig(X):
    Anow = X
    # k = 0
    test_quant = 1
    tol = 1e-6
    iter = 0
    max_iter = 100
    Q_all = np.eye(np.shape(X)[0])
    while iter < max_iter and test_quant > tol:
        # print(Anow)
        Q,R = qr_fact(copy.deepcopy(Anow))
        print(Q_all)
        Q_all = Q_all @ Q
        # Q, R = qr(Anow)
        Anext = R @ Q
        test_quant = np.linalg.norm(np.tril(Anext) - np.diag(np.diag(Anext)))
        Anow = Anext
        print(test_quant)
        iter += 1
        # print(Anow)
    # print(Anow)
    print(iter)
    # Q_all contains approximation of the eigenvectors only for symmetric X!!
    return Anow, Q_all
        
#%%
A = np.array([[16, 14, 14], [3, 11, 6], [-10, -16, -11]], dtype = 'float') #rnd.rand(4,4)
# A = np.array([[-10, 13, 13], [13, 7, -16], [13, -16, -7]], dtype = 'float')
# Q, R = qr(A)
# Q1, R1 = qr(A, mode='reduced')
Q2, R2 = qr_fact(copy.deepcopy(A))
print(A)
B,eigv = qr_eig(copy.deepcopy(A))
print(A)
# c,C = eig(A)

w, v = eig(A)
