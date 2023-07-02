# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:00:15 2023

@author: Jan-Philipp
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

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

def animate_VW(inp,t,eta):
    fps = 25
    nSeconds = 8

    snapshots_V = [ unpack_arr(np.abs(inp.T[i]),N)[1] for i in range( nSeconds * fps ) ]
    snapshots_W = [ unpack_arr(np.abs(inp.T[i]),N)[2] for i in range( nSeconds * fps ) ]
    
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
            #print( '.', end ='' )
            print(i/(fps*nSeconds)*100,"%completed\n")
    
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
    
    anim.save('test_anim,eta=%.3f_full.mp4'%eta, fps=fps, extra_args=['-vcodec', 'libx264'],dpi=300)
    
#animate_VW(inp,t)
PATH  = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=40,different etas_full_with_phi,phi=0.1\\"
for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    if not (eta == 0 or eta == -2 or eta==-10) and 2*eta%1==0 and eta > 0:
        print("Animating "+FILENAME)
        t_filename = "sol_full_t"+FILENAME[13:]
        path_inp = PATH+FILENAME
        path_t = PATH+t_filename
    
        
        inp = np.load(path_inp)
        t = np.load(path_t)
        
        animate_VW(inp,t,eta)
