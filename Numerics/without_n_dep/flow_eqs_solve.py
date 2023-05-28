# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:22:06 2023

@author: Jan-Philipp
"""
import numpy as np
from scipy.integrate import odeint

def flat_arr(om,V,W,eps):
    return np.append(om,np.append(np.append(V.flatten(),W.flatten()),eps))
def unpack_arr(flat,N):
    N = int(N)
    om0 = flat[0:N]
    V0 = flat[N:N+int(N**2)]
    W0 = flat[N+int(N**2):N+int(2*N**2)]
    eps = flat[-1]
    V = V0.reshape((N,N))
    W = W0.reshape((N,N))
    return om,V,W,eps

def deriv(flat,t):
    
N = 10 #the numer of modes
W = np.ones((N,N))*2
Wdag = np.conjugate(W)
V = np.ones((N,N))*3
om = np.ones(N)
eps = 0

flat = flat_arr(om,V,W,eps)
