# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 17:05:07 2023

@author: Jan-Philipp Christ
"""

import  quadratic_solver_wrapper as qs
import remove_linear_terms as rlt
import numpy as np
import matplotlib.pyplot as plt

eta = 1 #parameter which we will vary between -10...10
lambda_max = 20 #how far we want to traverse the flow
n = 200 #for how many points in the flow we want to save the Hamiltonian
N = 200
#grid = rlt.gen_1Dgrid(rlt.lamb_IR,rlt.lamb_UV,dk=(1e-1)/5)
#N = len(grid)
#sol = remove_linear(grid)

#om0,V0,W,eps = rlt.get_quadratic_Hamiltonian(grid)
om0,V0,W,eps = qs.unpack_arr(np.load('quadratic Hamiltonians N=200, different etas\\eta=8.182,N=200.npy'),200)
print(om0)

om = om0 + np.diag(V0)
V = V0 - np.diag(np.diag(V0))

#print("Successfully removed linear terms")

sol = qs.solve(om,V,W,eps,n,lambda_max, method = "RK45")

def eval(sol):
    sol2 = sol
    t = sol2["t"]
    for i in range(N):
        y = (sol2["y"][i])
        plt.plot(t,y)
    plt.title('omegas')
    plt.xlabel('\lambda')
    plt.show()

    for i in range(N,N+N**2):
        y = (sol2["y"][i])
        plt.plot(t,y)
    plt.title('V')
    plt.show()

    for i in range(N+N**2,N+2*N**2):
        y = (sol2["y"][i])
        plt.plot(t,y)
    plt.title('W')
    plt.show()

    y = np.abs(sol2["y"][-1])
    plt.plot(t,y)
    plt.title('epsilon')
    plt.show()
    om10,V10,W10,eps10 = qs.unpack_arr(sol2["y"].transpose()[n-1],N)

#eval(sol)
    