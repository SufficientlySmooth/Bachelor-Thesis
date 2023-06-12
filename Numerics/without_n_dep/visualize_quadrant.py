# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:20:18 2023

@author: Jan-Philipp
"""

import numpy as np
import  quadratic_solver_wrapper as qs
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True
})

N = 100

lambda_UV = 10
lambda_IR = .1
PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, one quadrant\\"


def visualize(sol,t,N,eta):
    om_end,V_end,W_end,eps_end = qs.unpack_arr(np.abs(inp.T[-1]),N)
    om_beg,V_beg,W_beg,eps_beg = qs.unpack_arr(np.abs(inp.T[0]),N)

    vmax_V = max(np.append(V_end.flatten(),V_beg.flatten()))
    vmax_W = max(np.append(W_end.flatten(),W_beg.flatten()))
    vmax = max(vmax_V,vmax_W)
    vmin_V = min(np.append(V_end.flatten(),V_beg.flatten()))
    vmin_W = min(np.append(W_end.flatten(),W_beg.flatten()))
    vmin = min(vmin_V,vmin_W)+1e-5
    print(vmin,vmax)
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    axes = axes.flatten()
    im1 = axes[0].imshow(V_beg, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,-lambda_IR,-lambda_UV,-lambda_IR])
    im2 = axes[1].imshow(V_end, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,-lambda_IR,-lambda_UV,-lambda_IR])
    im3 = axes[2].imshow(W_beg, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,-lambda_IR,-lambda_UV,-lambda_IR])
    im4 = axes[3].imshow(W_end, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,-lambda_IR,-lambda_UV,-lambda_IR])
    

    axes[2].set_xlabel(r"$k\xi$")
    axes[3].set_xlabel(r"$k\xi$")
    axes[1].set_title(r"$\lambda = %.2f$"%t[-1])
    axes[0].set_ylabel(r"$V$")
    axes[0].set_title(r"$\lambda = 0$")
    axes[2].set_ylabel(r"$W$")
    fig.suptitle(r"$\eta = %.3f$"%eta)
    
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    plt.savefig('V_W_imshow_plots\\eta=%.3f.pdf'%eta,dpi = 300)


for FILENAME in [file for file in os.listdir(PATH) if not "_t_" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    print(eta)
    if not eta == 0:
        t_filename = FILENAME[0:12]+"_t"+FILENAME[12:]
        path_inp = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, one quadrant\\"+FILENAME
        path_t = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, one quadrant\\"+t_filename
    
        
        inp = np.load(path_inp)
        t = np.load(path_t)
        
        visualize(inp,t,N,eta)
    