# -*- coding: utf-8 -*-
"""
Created on Tue May 30 18:15:20 2023

@author: Jan-Philipp
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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
    om, V, W, eps = unpack_arr(flat,N)
    Wdag = np.conjugate(W)
    
    om_ret =  np.array([np.sum([2*V[q,k]*V[k,q]*(om[k]-om[q])    -   2*(W[k,q]+W[q,k])*(om[k]+om[q])*(Wdag[q,k]+Wdag[k,q]) for q in range(N)]) for k in range(N)])
    V_ret = np.array([[-V[q,q_]*(om[q]-om[q_])**2 +   sum([-(W[q,p]+W[p,q])*(Wdag[p,q_]+Wdag[q_,p])*(om[q]+om[q_]+2*om[p])     +   V[p,q_]*V[q,p]*(om[q]+om[q_]-2*om[p]) 
                                         for p in range(N) if not p in (q,q_)]) 
                                 for q_ in range(N)] for q in range(N)])
    V_ret = V_ret * (1 - np.diag(np.ones(N)))
    W_ret = np.array([[-W[p,p_]*(om[p]+om[p_])**2 +   sum([-V[p,q]*(om[q]+om[p_])*(W[p_,q]+W[q,p_])    +   V[p,q]*(om[p]-om[q])*(W[q,p_]+W[p_,q]) for q in range(N) if not q==p])  for p_ in range(N)] for p in range(N)])
    eps_ret = -2*np.sum([   (W[p,p_]+W[p_,p])*(om[p]+om[p_])*Wdag[p,p_] 
                for p in range(N) for p_ in range(N)])
    return flat_arr(om_ret,V_ret,W_ret,eps_ret)

def eval(sol):
    sol2 = sol
    t = sol2["t"]
    for i in range(N):
        y = np.abs(sol2["y"][i])
        plt.plot(t,y)
    plt.title('omegas')
    plt.show()

    for i in range(N,N+N**2):
        y = np.abs(sol2["y"][i])
        plt.plot(t,y)
    plt.title('V')
    plt.show()

    for i in range(N+N**2,N+2*N**2):
        y = np.abs(sol2["y"][i])
        plt.plot(t,y)
    plt.title('W')
    plt.show()

    y = np.abs(sol2["y"][-1])
    plt.plot(t,y)
    plt.title('epsilon')
    plt.show()

    #print(sol2)

    om5,V5,W5,eps5 = unpack_arr(sol2["y"].transpose()[n//2],N)
    om10,V10,W10,eps10 = unpack_arr(sol2["y"].transpose()[n-1],N)
    print('Maximal absolute value of the Vs:', max(abs(V10.flatten())))
    print('Maximal absolute value of the Ws:', max(abs(W10.flatten())))
    print("Signature of V")
    print((abs(V10.flatten())<.001).reshape((N,N)))
    print("Signature of W")
    print((abs(W10.flatten())<.001).reshape((N,N)))

n = int(500) #number of t where the flow will be evaluated
tmax = 2 #we will calculate the flow until that point
N = 5 #the numer of modes
v_max = 2
W = (np.random.rand(N,N)-1/2)*v_max+(np.random.rand(N,N)-1/2)*v_max*1j#*np.array(list(range(N)))#np.ones((N,N))+(np.random.rand(N,N)-0.5)/2
#W = (W+np.transpose(W))/2
V = (np.random.rand(N,N)-1/2) * v_max * (1 - np.diag(np.ones(N))) #+ (np.random.rand(N,N)-1/2) * v_max * (1 - np.diag(np.ones(N)))*1j#*np.array(list(range(N)))

#V = (V+ np.transpose(V))/2
om = (np.random.rand(N)+1)*v_max#**2*(1+np.random.rand(N)/2)+np.pi
eps = 0

y0 = flat_arr(om,V,W,eps)


sol2 = solve_ivp(deriv,(0,tmax),y0,args=(N,),t_eval=np.linspace(0,tmax,int(n)),method='RK45')

eval(sol2)