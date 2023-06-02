# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:36:02 2023

@author: Jan-Philipp Christ
"""

import numpy as np
from scipy.integrate import solve_ivp

status = -1

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
    return om0,V,W,eps

def deriv(t,flat,N):
    global status
    om, V, W, eps = unpack_arr(flat,N)
    Wdag = np.conjugate(W)
    if int(t) > status:
        status = int(t)
        print("Current flow parameter: ",t)
    om_ret =  np.array([np.sum([2*V[q,k]*V[k,q]*(om[k]-om[q])    -   2*(W[k,q]+W[q,k])*(om[k]+om[q])*(Wdag[q,k]+Wdag[k,q]) for q in range(N)]) for k in range(N)])
    V_ret = np.array([[-V[q,q_]*(om[q]-om[q_])**2 +   sum([-(W[q,p]+W[p,q])*(Wdag[p,q_]+Wdag[q_,p])*(om[q]+om[q_]+2*om[p])     +   V[p,q_]*V[q,p]*(om[q]+om[q_]-2*om[p]) 
                                         for p in range(N) if not p in (q,q_)]) 
                                 for q_ in range(N)] for q in range(N)])
    V_ret = V_ret * (1 - np.diag(np.ones(N)))
    W_ret = np.array([[-W[p,p_]*(om[p]+om[p_])**2 +   sum([-V[p,q]*(om[q]+om[p_])*(W[p_,q]+W[q,p_])    +   V[p,q]*(om[p]-om[q])*(W[q,p_]+W[p_,q]) for q in range(N) if not q==p])  for p_ in range(N)] for p in range(N)])
    eps_ret = -2*np.sum([   (W[p,p_]+W[p_,p])*(om[p]+om[p_])*Wdag[p,p_] 
                for p in range(N) for p_ in range(N)])
    return flat_arr(om_ret,V_ret,W_ret,eps_ret)


def solve(om,V,W,eps,n,lambda_max, method = "RK45"):
    """
    Parameters:
    ---------------
    om: length n array with the omega_k
    V: NxN array of V_kk'
    W: NxN array of W_kk'
    eps: constant offset of the original Hamiltonian
    n: number of t where the flow will be evaluated
    lambda_max : we will calculate the flow until that point   
    ---------------
    returns sol object (see solve_ivp)
    """

    N = int(len(om))
    tmax = lambda_max
    eps = 0
    
    y0 = flat_arr(om,V,W,eps)
    
    
    return solve_ivp(deriv,(0,tmax),y0,args=(N,),t_eval=np.linspace(0,tmax,int(n)),method=method)
