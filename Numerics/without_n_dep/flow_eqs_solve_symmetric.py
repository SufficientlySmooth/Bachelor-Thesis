# -*- coding: utf-8 -*-
"""
Created on Sun May 28 14:22:06 2023

@author: Jan-Philipp
"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from numba import jit

@jit
def flat_arr(om,V,W,eps):
    return np.append(om,np.append(np.append(V.flatten(),W.flatten()),eps))

@jit
def unpack_arr(flat,N):
    N = int(N)
    om0 = flat[0:N]
    V0 = flat[N:N+int(N**2)]
    W0 = flat[N+int(N**2):N+int(2*N**2)]
    eps = flat[-1]
    V = V0.reshape((N,N))
    W = W0.reshape((N,N))
    return om0,V,W,eps

@jit
def deriv(t,flat,N):
    om, V, W, eps = unpack_arr(flat,N)
    Wdag = np.conjugate(W)
    
    om_ret = np.array([np.sum([2*V[q,k]*V[k,q]*(om[k]-om[q])    -   2*(W[k,q]+W[q,k])*(om[k]+om[q])*(Wdag[q,k]+Wdag[k,q]) for q in range(N)]) for k in range(N)])

    V_ret = np.array(
        [[-V[q,q_]*(om[q]-om[q_])**2    +   sum([-(W[q,p]+W[p,q])*(Wdag[p,q_]+Wdag[q_,p])*(om[q]+om[q_]+2*om[p])     +   V[p,q_]*V[q,p]*(om[q]+om[q_]-2*om[p]) 
                                             for p in range(N)]) 
         for q_ in range(N)] 
         for q in range(N)]
        )
    V_ret = V_ret * (1 - np.diag(np.ones(N)))
    W_ret = np.array(
        [[  -W[p,p_]*(om[p]+om[p_])**2      +   sum([-V[p,q]*(om[q]+om[p_])*(W[p_,q]+W[q,p_])    +   V[p,q]*(om[p]-om[q])*(W[q,p_]+W[p_,q]) for q in range(N)]) 
          for p_ in range(N)] 
         for p in range(N)]
        )
    
    eps_ret = -2*np.sum([   (W[p,p_]+W[p_,p])*(om[p]+om[p_])*Wdag[p,p_] 
                for p in range(N) for p_ in range(N)])
    
    return flat_arr(om_ret,(V_ret+ np.transpose(V_ret))/2,W_ret,eps_ret)


n = int(200) #number of t where the flow will be evaluated
tmax = .1 #we will calculate the flow until that point
N = 11 #the numer of modes
W = np.random.rand(N,N)*14-10#*np.array(list(range(N)))#np.ones((N,N))+(np.random.rand(N,N)-0.5)/2
W = (W+np.transpose(W))/2
V = np.random.rand(N,N)*14-10#*np.array(list(range(N)))
for i in range(N):
    V[i,i]=0
V = (V+ np.transpose(V))/2
om = np.array(list(range(N)))**2*(1+np.random.rand(N)/2)+10
eps = 0

flat = flat_arr(om,V,W,eps)

y0 = flat_arr(om,V,W,eps)

sol2 = solve_ivp(deriv,(0,tmax),y0,args=(N,),t_eval=np.linspace(0,tmax,int(n)),method='Radau')

t = sol2["t"]
for i in range(N):
    y = sol2["y"][i]
    plt.plot(t,y)
plt.title('omegas')
plt.show()

for i in range(N,N+N**2):
    y = sol2["y"][i]
    plt.plot(t,y)
plt.title('V')
plt.show()

for i in range(N+N**2,N+2*N**2):
    y = sol2["y"][i]
    plt.plot(t,y)
plt.title('W')
plt.show()

y = sol2["y"][-1]
plt.plot(t,y)
plt.title('epsilon')
plt.show()

print(sol2)

om5,V5,W5,eps5 = unpack_arr(sol2["y"].transpose()[n//2],N)
om10,V10,W10,eps10 = unpack_arr(sol2["y"].transpose()[n-1],N)
print('Maximal absolute value of the Vs:', max(abs(V10.flatten())))
print('Maximal absolute value of the Ws:', max(abs(W10.flatten())))

print((abs(V10.flatten())<.001).reshape((N,N)))


#sol = odeint(deriv, y0, np.linspace(0,100,1000),args=(N,),tfirst=True,full_output=1)
#om_res, V_res, W_res, eps_res = unpack_arr(sol2["y"],N)