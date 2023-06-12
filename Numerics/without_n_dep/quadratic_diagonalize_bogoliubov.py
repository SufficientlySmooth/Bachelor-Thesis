# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 15:16:21 2023

@author: Jan-Philipp
"""

import numpy as np
import  quadratic_solver_wrapper as qs
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy
plt.rcParams.update({
    "text.usetex": True
})

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

eta = 1 #parameter which we will vary between -10...10
lambda_max = 40 #how far we want to traverse the flow
n = 200 #for how many points in the flow we want to save the Hamiltonian
#grid = rlt.gen_1Dgrid(rlt.lamb_IR,rlt.lamb_UV,dk=(1e-1)/5)
#N = len(grid)
#sol = remove_linear(grid)

#om0,V0,W,eps = rlt.get_quadratic_Hamiltonian(grid)


N = 200

lambda_UV = 10


#om0,V0,W,eps = qs.unpack_arr(np.load("C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit Quadratic Hamiltonians\\Ham_eta=1.200000000000001,N=200,lambda_IR=0.1,lambda_UV=10.0.npy"),100)

#V = V0 + np.diag(np.diag(om0)) # - np.diag(np.diag(V0)) #


#A = V
#B = W



Omega = np.block([[np.diag(np.ones(N)),np.zeros((N,N))],[np.zeros((N,N)),-np.diag(np.ones(N))]])
theta = np.block([[np.zeros((N,N)),np.diag(np.ones(N))],[np.diag(np.ones(N)),np.zeros((N,N))]])

#print(eigvals)

def get_eigvals(A,B):
    H = np.block([[A,B],[-np.conj(B),-np.conj(A)]])
    eigvals, eigvecs = scipy.linalg.eig(H)
    eigvecs = eigvecs.T
    sorted_vals = np.array([val for _, val in sorted(zip(eigvals,eigvals),key = lambda tup: np.real(tup[0]))])#np.sort(np.real(eigvals))
    sorted_vecs = np.array([vec for _, vec in sorted(zip(eigvals,eigvecs),key = lambda tup: np.real(tup[0]))])

    eigvals_ord = np.array([[sorted_vals[i],sorted_vals[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    eigvecs_ord = np.array([[sorted_vecs[i],sorted_vecs[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    
    return eigvals_ord, eigvecs_ord

def find_right_eigvals(eigvals_ord,eigvecs_ord):
    #a = [] #contains 0 at position N if eigvals[N,0] is the right candidate, 1 at position N if eigvals[N,1] is the right candidate
    right_eigvals = []
    for i, vecs in enumerate(eigvecs_ord):
        vec0 = vecs[0]
        vec1 = vecs[1] #np.isclose(eigvecs_ord[j,0],np.conj(theta@eigvecs_ord[j,1])) should be true!!!
        matrix_element_0 = np.conj(vec0.T)@Omega@vec0
        matrix_element_1 = np.conj(vec1.T)@Omega@vec1
        #print(matrix_element_0,matrix_element_1)
        if (matrix_element_0>0) and (matrix_element_1<0):
            right_eigvals.append(eigvals_ord[i][0])
        elif (matrix_element_1>0) and (matrix_element_0<0):           
            right_eigvals.append(eigvals_ord[i][1])
        else:
            if not (np.isclose(matrix_element_1,0) and np.isclose(matrix_element_0,0)):
                if not np.abs(eigvals_ord[i][0])<1e-10:
                    print("First matrix element: ",matrix_element_0,"\n Second matrix element: ",matrix_element_1," \n")
                    raise ValueError("Both Matrix Elements have the same sign and are not small!!!")
            else:
                print("There were matrix elements with the same sign, however they were so small that they can be neglected")
    return right_eigvals

def ground_state_energy(right_eigvals,E_0,A):
    return E_0-1/2*np.sum(np.diag(A))+1/2*np.sum(right_eigvals) #see eq. 33 in Practial Course manual



import os
PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit Quadratic Hamiltonians\\"


etas = []
epss = []
for FILENAME in [file for file in os.listdir(PATH) if not "_t_" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    path_inp = PATH+FILENAME
    om0,V0,W,eps = qs.unpack_arr(np.load(path_inp),200)
    V = V0 + np.diag(np.diag(om0)) # - np.diag(np.diag(V0)) #
    
    A = V
    B = W
  
    eigvals_ord, eigvecs_ord = get_eigvals(A,B)
    right_eigvals = find_right_eigvals(eigvals_ord,eigvecs_ord)
    #print(right_eigvals)
    print("Maximal right eigenvalue: ", max(np.abs(right_eigvals)))
    etas.append(eta)
    epss.append(np.real(ground_state_energy(right_eigvals,eps,A)))

plt.plot(etas,epss,linestyle='None',marker='x',color='black',label=r'via Bogliubov Transformation')
plt.xlabel(r'$\eta=g_{IB}/g_{BB}$')
plt.ylabel(r'$E_0$')
plt.legend()
plt.savefig('E_0-eta_via_Bogliubov_Trafo.pdf',dpi=300)
plt.show()
