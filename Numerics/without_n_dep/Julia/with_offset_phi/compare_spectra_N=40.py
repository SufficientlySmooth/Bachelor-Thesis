# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:14:53 2023

@author: Jan-Philipp
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib 
matplotlib.style.use('JaPh')

"""
plt.rcParams.update({
    "text.usetex": True
})

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
"""

N = 40


PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=40,different etas_full_with_phi,phi=0.1\\"#N=40,lambda_IR=0.1+phi,lmabda_UV=10+phi\\"
PATH_BOG = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit Quadratic Hamiltonians\\N=40,lambda_IR=0.1+phi,lambda_UV=10+phi,phi=0.1\\"

etas = []
epss = []
epss_err = []
Omega = np.block([[np.diag(np.ones(N)),np.zeros((N,N))],[np.zeros((N,N)),-np.diag(np.ones(N))]])
theta = np.block([[np.zeros((N,N)),np.diag(np.ones(N))],[np.diag(np.ones(N)),np.zeros((N,N))]])

def unpack_arr(flat,N):
    N = int(N)
    om0 = flat[0:N]
    V0 = flat[N:N+int(N**2)]
    W0 = flat[N+int(N**2):N+int(2*N**2)]
    eps = flat[-1]
    V = V0.reshape((N,N))
    W = W0.reshape((N,N))
    return om0,V,W,eps

def get_eigvals(A,B):
    H = np.block([[A,B],[-np.conj(B),-np.conj(A)]])
    eigvals, eigvecs = np.linalg.eig(H)
    eigvecs = eigvecs.T
    if np.isclose(np.linalg.det(eigvecs),0):
        ValueError("Eigenvectors are not linearly independent")
    else:
        print("Determinant of all eigenvectors is ", np.linalg.det(eigvecs))
    sorted_vals = np.array([val for _, val in sorted(zip(eigvals,eigvals),key = lambda tup: np.real(tup[0]))])#np.sort(np.real(eigvals))
    sorted_vecs = np.array([vec for _, vec in sorted(zip(eigvals,eigvecs),key = lambda tup: np.real(tup[0]))])
        
    eigvals_ord = np.array([[sorted_vals[i],sorted_vals[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    eigvecs_ord = np.array([[sorted_vecs[i],sorted_vecs[len(eigvals)-i-1]] for i in range(len(eigvals)//2)])
    
    return eigvals_ord, eigvecs_ord

def find_right_eigvals(eigvals_ord,eigvecs_ord):
    imag_eigvals = np.zeros(len(eigvals_ord))
    right_eigvals = []
    conj_eigvals = []
    for i, vecs in enumerate(eigvecs_ord):
        vec0 = vecs[0]
        vec1 = vecs[1] 
        #print(eigvals_ord[i])
        matrix_element_0 = np.conj(vec0.T)@Omega@vec0
        matrix_element_1 = np.conj(vec1.T)@Omega@vec1
        if (matrix_element_0>0) and (matrix_element_1<0):# and not np.abs(np.imag(eigvals_ord[i][0]))<1e-10:
            right_eigvals.append(eigvals_ord[i][0])
            conj_eigvals.append(-eigvals_ord[i][1])
        elif (matrix_element_1>0) and (matrix_element_0<0):# and not np.abs(np.imag(eigvals_ord[i][1]))<1e-10:           
            right_eigvals.append(eigvals_ord[i][1])
            conj_eigvals.append(-eigvals_ord[i][0])
        else:
            right_eigvals.append(0)
            conj_eigvals.append(0)
            if np.isclose(np.real(eigvals_ord[i][0]),0):
                imag_eigvals[i] = np.abs(np.imag(eigvals_ord[i][1]))
            else:
                ValueError("Complex Eigenvalue which is not purely imaginary!")
            
    return imag_eigvals, right_eigvals, conj_eigvals


def get_plot_data_bogoliubov(V0, W0, om0):
    V = V0 + np.diag(om0) # - np.diag(np.diag(V0)) #
    W = 1/2 * (W0 + W0.T)
    #print("eta=",eta,"eps=",eps)
    A = V
    B = 2*W
    eigvals_ord, eigvecs_ord = get_eigvals(A,B)
    imag_eigvals, right_eigvals, conj_eigvals = find_right_eigvals(eigvals_ord,eigvecs_ord)
        
    return right_eigvals


eigvals_bog = []
eigvals_flow = []
eigvals_flow_and_bog = []

for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    path_inp = PATH+FILENAME
    path_t = PATH+t_filename

    inp = np.load(path_inp)
    om0_start,V0_start,W0_start,eps_start = unpack_arr(inp.T[0],N)
    om0_end,V0_end,W0_end,eps_end = unpack_arr(inp.T[-1],N)

    right_eigvals_bog_and_flow = get_plot_data_bogoliubov(V0_end, W0_end, om0_end)
    right_eigvals_flow = om0_end
    
    etas.append(eta)
    
    eigvals_flow.append(np.sort(right_eigvals_flow))
    eigvals_flow_and_bog.append(np.sort(right_eigvals_bog_and_flow))

etas_bog = []

for FILENAME in [file for file in os.listdir(PATH_BOG) if not "_t_" in file]:
    eta_bog = float(FILENAME.split(',')[0].split('=')[-1])

    path_inp = PATH_BOG+FILENAME
    inp = np.load(path_inp)
    om0_start,V0_start,W0_start,eps_start = unpack_arr(inp,N)
    right_eigvals_bog = get_plot_data_bogoliubov(V0_start, W0_start, om0_start)
       
    etas_bog.append(eta_bog)
    eigvals_bog.append(np.sort(right_eigvals_bog))
    
    
    

eta_filter_bog = np.abs(np.array(etas_bog)) < 14
eta_sort_filter_bog = np.argsort(etas_bog)
etas_bog = np.array(etas_bog)[eta_filter_bog]
eigvals_bog = np.array(eigvals_bog)[eta_filter_bog]

fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) 

ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)


ax.plot(etas,[eigvals[-1] for eigvals in eigvals_flow],marker='x',color = 'lime',markersize=4,linestyle='None',label = 'via flow')
#ax.plot(etas,[eigvals[-1] for eigvals in eigvals_flow_and_bog],marker='+',color='firebrick',markersize=4,linestyle='None',label = 'via flow and Bogoliubov')
ax.plot(etas_bog,[eigvals[-1] for eigvals in eigvals_bog],marker='o',color='steelblue',markersize=3,linestyle='None',label = 'via Bogoliubov')

ax.plot(etas,[eigvals[-2] for eigvals in eigvals_flow],marker='+',color = 'lime',markersize=4,linestyle='None',label = 'via flow')
#ax.plot(etas,[eigvals[-2] for eigvals in eigvals_flow_and_bog],marker='o',color='firebrick',markersize=4,linestyle='None')#,label = r'$2^{\mathrm{nd}}$ largest $\omega$ flow and Bogoliubov')
ax.plot(etas_bog,[eigvals[-2] for eigvals in eigvals_bog],marker='.',color='steelblue',markersize=3,linestyle='None',label = 'via Bogoliubov')


for i in range(0,5):
    ax.plot(etas,[eigvals[-3-2*i] for eigvals in eigvals_flow],marker='x',color = 'lime',markersize=4,linestyle='None')
    #ax.plot(etas,[eigvals[-3-2*i] for eigvals in eigvals_flow_and_bog],marker='+',color='firebrick',markersize=4,linestyle='None')
    ax.plot(etas_bog,[eigvals[-3-2*i] for eigvals in eigvals_bog],marker='o',color='steelblue',markersize=3,linestyle='None')
    
    ax.plot(etas,[eigvals[-3-2*i-1] for eigvals in eigvals_flow],marker='+',color = 'lime',markersize=4,linestyle='None')
    #ax.plot(etas,[eigvals[-3-2*i-1] for eigvals in eigvals_flow_and_bog],marker='o',color='firebrick',markersize=4,linestyle='None')
    ax.plot(etas_bog,[eigvals[-3-2*i-1] for eigvals in eigvals_bog],marker='.',color='steelblue',markersize=3,linestyle='None')

"""
ax.plot(etas,[eigvals[-31] for eigvals in eigvals_flow],marker='x',color = 'lime',markersize=4,linestyle='None',label = 'via flow')
#ax.plot(etas,[eigvals[-1] for eigvals in eigvals_flow_and_bog],marker='+',color='firebrick',markersize=4,linestyle='None',label = 'via flow and Bogoliubov')
ax.plot(etas_bog,[eigvals[-31] for eigvals in eigvals_bog],marker='o',color='steelblue',markersize=3,linestyle='None',label = 'via Bogoliubov')

ax.plot(etas,[eigvals[-32] for eigvals in eigvals_flow],marker='+',color = 'lime',markersize=4,linestyle='None',label = 'via flow')
#ax.plot(etas,[eigvals[-2] for eigvals in eigvals_flow_and_bog],marker='o',color='firebrick',markersize=4,linestyle='None')#,label = r'$2^{\mathrm{nd}}$ largest $\omega$ flow and Bogoliubov')
ax.plot(etas_bog,[eigvals[-32] for eigvals in eigvals_bog],marker='.',color='steelblue',markersize=3,linestyle='None',label = 'via Bogoliubov')


for i in range(15,19):
    ax.plot(etas,[eigvals[-3-2*i] for eigvals in eigvals_flow],marker='x',color = 'lime',markersize=4,linestyle='None')
    #ax.plot(etas,[eigvals[-3-2*i] for eigvals in eigvals_flow_and_bog],marker='+',color='firebrick',markersize=4,linestyle='None')
    ax.plot(etas_bog,[eigvals[-3-2*i] for eigvals in eigvals_bog],marker='o',color='steelblue',markersize=3,linestyle='None')
    
    ax.plot(etas,[eigvals[-3-2*i-1] for eigvals in eigvals_flow],marker='+',color = 'lime',markersize=4,linestyle='None')
    #ax.plot(etas,[eigvals[-3-2*i-1] for eigvals in eigvals_flow_and_bog],marker='o',color='firebrick',markersize=4,linestyle='None')
    ax.plot(etas_bog,[eigvals[-3-2*i-1] for eigvals in eigvals_bog],marker='.',color='steelblue',markersize=3,linestyle='None')
"""

ax.tick_params(axis='both',labelsize=10)
ax.set_xlabel(r'$\eta$',fontsize = 16)
ax.set_ylabel(r'Eigenenergies $\omega[c/\xi]$',fontsize = 16)
ax.set_xlim(-10.5,10.5)


ax.legend(loc='best',fontsize=14,ncols=2)
plt.tight_layout()
plt.savefig('Spectral_analysis,N=40.pdf',dpi=300)
plt.show()