# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 21:21:34 2023

@author: Jan-Philipp
"""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import  quadratic_solver_wrapper as qs
import os
import matplotlib.animation as animation

plt.rcParams.update({
    "text.usetex": True
})

N = 200
#eta = 10
lambda_UV = 10


path_inp = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\sol_quadrant_Ham_eta=10.0,N=200,lambda_IR=0.1,lambda_UV=10.0.npy"
path_t = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\sol_full_tHam_eta=10.0,N=200,lambda_IR=0.1,lambda_UV=10.0.npy"
#path_inp = "C:\\Users\\Jan-Philipp\\sol_01.npy"
#path_t = "C:\\Users\\Jan-Philipp\\sol_01_t.npy"

inp = np.load(path_inp)
t = np.load(path_t)

def animate_VW(inp,t,eta):
    fps = 50
    nSeconds = 10

    snapshots_V = [ qs.unpack_arr(np.abs(inp.T[i]),N)[1] for i in range( nSeconds * fps ) ]
    snapshots_W = [ qs.unpack_arr(np.abs(inp.T[i]),N)[2] for i in range( nSeconds * fps ) ]
    
    vmax = max(np.append(snapshots_V[0].flatten(),snapshots_W[0].flatten()))    
    fig, axes = plt.subplots(nrows=1, ncols=2)
        
    axes = axes.flatten()
    
    im1 = axes[0].imshow(snapshots_V[0], aspect='auto', cmap='binary',vmin = 1e-5, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    im2 = axes[1].imshow(snapshots_W[0], aspect='auto', cmap='binary',vmin = 1e-5, vmax = vmax,norm='log',extent = [-lambda_UV,lambda_UV,-lambda_UV,lambda_UV])
    
    axes[0].set_xlabel(r"$k\xi$")
    axes[1].set_xlabel(r"$k\xi$")
    axes[0].set_ylabel(r"$k\xi$")
    #axes[1].set_ylabel(r"$k\xi$")
    
    axes[0].set_title(r"$V[c/\xi]$")
    axes[1].set_title(r"$W[c/\xi]$")
    fig.suptitle(r"$\eta=%.3f,\lambda = %.2f$"%(eta,0))
    
    plt.colorbar(im2, extend='max')
    
    def animate_func(i):
        if i % fps == 0:
            print( '.', end ='' )
    
        im1.set_array(snapshots_V[i])
        im2.set_array(snapshots_W[i])
        fig.suptitle(r"$\eta=%.3f,\lambda = %.2f$"%(eta,t[i]))
        return [im1,im2]
    
    anim = animation.FuncAnimation(
                                   fig, 
                                   animate_func, 
                                   frames = nSeconds * fps,
                                   interval = 500 / fps, # in ms
                                   )
    
    anim.save('test_anim,eta=%.3f_full.mp4'%eta, fps=fps, extra_args=['-vcodec', 'libx264'])
    
#animate_VW(inp,t)
PATH  = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas, full, V_diag is zero\\"
for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    print("Animating "+FILENAME)
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    path_inp = PATH+FILENAME
    path_t = PATH+t_filename

    
    inp = np.load(path_inp)
    t = np.load(path_t)
    
    animate_VW(inp,t,eta)