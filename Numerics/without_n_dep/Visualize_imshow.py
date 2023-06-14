# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 16:21:34 2023

@author: Jan-Philipp
"""
import numpy as np
import  quadratic_solver_wrapper as qs
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True
})


eta = 10
N = 200
path_inp = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\sol_quadrant_Ham_eta=10.0,N=200,lambda_IR=0.1,lambda_UV=10.0.npy"
path_t = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\sol_full_tHam_eta=10.0,N=200,lambda_IR=0.1,lambda_UV=10.0.npy"
#path_inp = "C:\\Users\\Jan-Philipp\\sol_01.npy"
#path_t = "C:\\Users\\Jan-Philipp\\sol_01_t.npy"

inp = np.load(path_inp)
t = np.load(path_t)
lambda_UV = 10




def visualize(sol,t,N,eta):
    om_end,V_end,W_end,eps_end = qs.unpack_arr(np.abs(inp.T[-1]),N)
    om_beg,V_beg,W_beg,eps_beg = qs.unpack_arr(np.abs(inp.T[0]),N)

    vmax_V = max(np.append(V_end.flatten(),V_beg.flatten()))
    vmax_W = max(np.append(W_end.flatten(),W_beg.flatten()))
    vmax = max(vmax_V,vmax_W)
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    axes = axes.flatten()
    im1 = axes[0].imshow(V_beg, aspect='auto', cmap='binary',vmin = 1e-5, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    im2 = axes[1].imshow(V_end, aspect='auto', cmap='binary',vmin = 1e-5, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    im3 = axes[2].imshow(W_beg, aspect='auto', cmap='binary',vmin = 1e-5, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    im4 = axes[3].imshow(W_end, aspect='auto', cmap='binary',vmin = 1e-5, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    

    axes[2].set_xlabel(r"$k\xi$")
    axes[3].set_xlabel(r"$k\xi$")
    axes[1].set_title(r"$\lambda = %.2f$"%t[-1])
    axes[0].set_ylabel(r"$V[c/\xi]$")
    axes[0].set_title(r"$\lambda = 0$")
    axes[2].set_ylabel(r"$W[c/\xi]$")
    #fig.suptitle(r"$\eta = %.3f$"%eta)
    
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    plt.savefig('eta=%.3f_full.pdf'%eta,dpi = 300)

#visualize(inp,t,N)

#animate_VW(inp,t)
PATH  = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\"
for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    path_inp = PATH+FILENAME
    path_t = PATH+t_filename

    
    inp = np.load(path_inp)
    t = np.load(path_t)
    try:
        visualize(inp,t,N,eta)
    except:
        continue