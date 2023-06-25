# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 13:24:38 2023

@author: Jan-Philipp
"""

import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "text.usetex": True
})


N = 40
lambda_UV = 10


def unpack_arr(flat,N):
    N = int(N)
    om0 = flat[0:N]
    V0 = flat[N:N+int(N**2)]
    W0 = flat[N+int(N**2):N+int(2*N**2)]
    eps = flat[-1]
    V = V0.reshape((N,N))
    W = W0.reshape((N,N))
    return om0,V,W,eps

def visualize(sol,t,N,eta):
    om_end,V_end,W_end,eps_end = unpack_arr(np.abs(inp.T[-1]),N)
    om_beg,V_beg,W_beg,eps_beg = unpack_arr(np.abs(inp.T[0]),N)

    vmax_V = max(np.append(V_end.flatten(),V_beg.flatten()))
    vmax_W = max(np.append(W_end.flatten(),W_beg.flatten()))
    vmax = max(vmax_V,vmax_W)
    #vmin_V = min(np.append(V_end.flatten()[V_end.flatten()>1e-10],V_beg.flatten()[V_beg.flatten()>1e-10]))
    #vmin_W = min(np.append(W_end.flatten()[W_end.flatten()>1e-10],W_beg.flatten()[W_beg.flatten()>1e-10]))
    #vmin = min(vmin_V,vmin_W)
    #print(vmin)
    vmin = 1e-5
    
    fig, axes = plt.subplots(nrows=2, ncols=2)
    
    axes = axes.flatten()
    im1 = axes[0].imshow(V_beg, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    im2 = axes[1].imshow(V_end, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    im3 = axes[2].imshow(W_beg, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    im4 = axes[3].imshow(W_end, aspect='auto', cmap='binary',vmin = vmin, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    
    for axis in axes:
        axis.grid(visible=False)

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
    plt.savefig('visualize_imshow_N=40,phi=0.1\\eta=%.3f_full.pdf'%eta,dpi = 300)

#visualize(inp,t,N)

#animate_VW(inp,t)
PATH  = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=40,different etas_full_with_phi,phi=0.1\\"
for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    if not eta == 0:
        t_filename = "sol_full_t"+FILENAME[13:]
        path_inp = PATH+FILENAME
        path_t = PATH+t_filename
    
        
        inp = np.load(path_inp)
        t = np.load(path_t)
        visualize(inp,t,N,eta)
