# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 20:22:57 2023

@author: Jan-Philipp
"""
import  quadratic_solver_wrapper as qs
import remove_linear_terms as rlt
import numpy as np
import matplotlib.pyplot as plt
import scipy

N = 100

m_0 = 42

m_1 = 1e-5

w = 6*np.pi * np.random.rand(N)*1000


alpha = m_0*w/m_1*1/4
beta = m_1*w/m_0*1/4

eps = np.sum(alpha + beta)
V = np.diag((alpha+beta))
W = np.diag((beta-alpha))

Omega = np.block([[np.diag(np.ones(N)),np.zeros((N,N))],[np.zeros((N,N)),-np.diag(np.ones(N))]])
theta = np.block([[np.zeros((N,N)),np.diag(np.ones(N))],[np.diag(np.ones(N)),np.zeros((N,N))]])


def get_eigvals(A,B):
    H = np.block([[A,B],[-np.conj(B),-np.conj(A)]])
    eigvals, eigvecs = np.linalg.eig(H)
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

eigvals_ord, eigvecs_ord = get_eigvals(V,W)
right_eigvals = find_right_eigvals(eigvals_ord,eigvecs_ord)

E_0 = ground_state_energy(right_eigvals,eps,V)

print(np.isclose(np.sort(right_eigvals)*2,np.sort(w),rtol = 1e-2))