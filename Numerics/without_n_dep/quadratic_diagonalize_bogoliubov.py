# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:16:21 2023

@author: Jan-Philipp
"""

import numpy as np
import  quadratic_solver_wrapper as qs
import remove_linear_terms as rlt
import matplotlib.pyplot as plt

eta = 1 #parameter which we will vary between -10...10
lambda_max = 20 #how far we want to traverse the flow
n = 200 #for how many points in the flow we want to save the Hamiltonian
N = 200
#grid = rlt.gen_1Dgrid(rlt.lamb_IR,rlt.lamb_UV,dk=(1e-1)/5)
#N = len(grid)
#sol = remove_linear(grid)

#om0,V0,W,eps = rlt.get_quadratic_Hamiltonian(grid)


N = 200
eta = -0.707
lambda_UV = 10


om0,V0,W,eps = qs.unpack_arr(np.load('quadratic Hamiltonians N=200, different etas\\eta=%.3f,N=200.npy'%eta),200)

V = V0 + np.diag(np.diag(om0))

A = V
B = W
H = np.block([[A,B],[-np.conj(B),-np.conj(A)]])
Omega = np.block([[np.ones((N,N)),np.zeros((N,N))],[np.zeros((N,N)),-np.ones((N,N))]])
eigvals, eigvecs = np.linalg.eig(H)
print(eigvals)



path_inp = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas\\sol_full_eta=%.3f,N=200.npy"%eta
inp = np.load(path_inp)
om_end,V_end,W_end,eps_end = qs.unpack_arr(np.abs(inp.T[-1]),N)


om_bog = om_bog = np.sort(np.abs(np.real(eigvals)))
plt.plot(list(range(len(om_end))),np.sort(np.abs(om_end)))
plt.plot(list(range(int(len(om_bog)/2))),om_bog[::2])
plt.show()

"""
pos = []

for i in range(len(eigvals)):
    vec = eigvecs[i]
    #print(np.conj(vec.T)@Omega@vec)
    mel = np.conj(vec.T)@Omega@vec
    print(mel)
    if mel>0 and not np.isclose(mel,0):
        pos.append(eigvals[i])

print(pos)"""

