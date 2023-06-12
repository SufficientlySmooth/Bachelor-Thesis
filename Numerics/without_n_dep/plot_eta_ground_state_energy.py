# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:14:06 2023

@author: Jan-Philipp
"""

import numpy as np
import  quadratic_solver_wrapper as qs
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True
})

N = 200

lambda_UV = 10
lambda_IR = .1
PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\"


etas = []
epss = []
for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    path_inp = PATH+FILENAME
    path_t = PATH+t_filename

    
    inp = np.load(path_inp)
    eps_eta = inp[-1][-1]
    
    etas.append(eta)
    epss.append(eps_eta)

plt.plot(etas,epss,linestyle='None',marker='x')