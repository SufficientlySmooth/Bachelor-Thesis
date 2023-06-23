# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:53:25 2023

@author: Jan-Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

matplotlib.style.use('JaPh')
N = 40

lambda_UV = 10
lambda_IR = .1
PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=40,different etas_full_with_phi\\"

#etas_bog,epss_bog,epss_err_bog = np.load('eta_E_0_bog.npy')
#etas_flow_and_bog,epss_flow_and_bog,epss_err_flow_and_bog = np.load('eta_E_0_flow_and_bog.npy')
etas = []
epss = []
minimal_omega = []
smallest_positive_omega = []
for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    path_inp = PATH+FILENAME
    path_t = PATH+t_filename

    
    inp = np.load(path_inp)
    eps_eta = inp.T[-1][-1]
    om = inp.T[-1][0:N]
    minimal_omega.append(min(om))
    if eta == 7:
        print(om[om>0])
    smallest_positive_omega.append(min(om[om>0]))
    
    etas.append(eta)
    epss.append(eps_eta)

plt.plot(etas,epss,linestyle='None',marker='x',label=r'via Flow Equations')
#plt.errorbar(etas_bog,np.real(epss_bog),yerr = np.real(epss_err_bog),linestyle='None',marker=',',color='black',label=r'via Bogoliubov Transformation',capsize = 1.5, ecolor = 'red')
#plt.errorbar(etas_flow_and_bog,np.real(epss_flow_and_bog),yerr = np.real(epss_err_flow_and_bog),linestyle='None',marker='+',label=r'Bogoliubov Transformation after Flow Equations',capsize = 1.5, ecolor = 'orange')
plt.xlabel(r'$\eta=g_{IB}/g_{BB}$',fontsize=14)
plt.ylabel(r'$E_0[c/\xi]$',fontsize=14)
plt.xlim(min(etas)-1,max(etas)+1)
plt.ylim(min(epss)*1.1,max(epss)*1.1)
plt.legend(loc = 'best',fontsize=10)
plt.tick_params(axis='both', which='minor')
plt.grid(visible=True, which='both', color="grey", linestyle='-', alpha=.2,linewidth=.008)
plt.savefig('E_0-eta_via_Flow_Eqs,N=%i.pdf'%N,dpi=300)
plt.show()

plt.clf()
plt.plot(etas,minimal_omega,marker='.',linestyle='None',label=r'minimal $\omega_k$ after flow before Bogoliubov Trafo')
plt.plot(etas,smallest_positive_omega,marker='.',linestyle='None',label=r'smallest positive $\omega_k$ after flow before Bogoliubov Trafo')
plt.xlabel(r'$\eta=g_{IB}/g_{BB}$',fontsize=14)
plt.ylabel(r'$\omega[c/\xi]$',fontsize=14)
plt.xlim(min(etas)-1,max(etas)+1)
plt.legend(fontsize=10,loc='best')
plt.savefig('spectrum_different_etas_N=%i.pdf'%N,dpi=300)
plt.show()