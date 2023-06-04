# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:16:21 2023

@author: Jan-Philipp
"""

import numpy as np

N = 3
v_max = 5
A = (np.random.rand(N,N)-1/2)*v_max+(np.random.rand(N,N)-1/2)*v_max*1j
B = (np.random.rand(N,N)-1/2)*v_max+(np.random.rand(N,N)-1/2)*v_max*1j
H = np.block([[A,B],[-np.conj(B),-np.conj(A)]])
Omega = np.block([[np.ones((N,N)),np.zeros((N,N))],[np.zeros((N,N)),-np.ones((N,N))]])
eigvals, eigvecs = np.linalg.eig(H)

pos = []

for i in range(len(eigvals)):
    vec = eigvecs[i]
    #print(np.conj(vec.T)@Omega@vec)
    mel = np.conj(vec.T)@Omega@vec
    print(mel)
    if mel>0 and not np.isclose(mel,0):
        pos.append(eigvals[i])

print(pos)

