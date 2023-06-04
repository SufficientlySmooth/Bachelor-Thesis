# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 00:32:02 2023

@author: Jan-Philipp
"""

import numpy as np
import  quadratic_solver_wrapper as qs
import matplotlib.pyplot as plt
import matplotlib.colors as colors

N = 100
path_inp = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas\\sol_eta=1.111,N=200.npy"
path_t = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=200,different etas\\sol_teta=1.111,N=200.npy"
#path_inp = "C:\\Users\\Jan-Philipp\\sol_01.npy"
#path_t = "C:\\Users\\Jan-Philipp\\sol_01_t.npy"

inp = np.load(path_inp)
t = np.load(path_t)

"""
for i in range(200,40200):
    plt.plot(t,inp[i],linewidth=1)
#plt.ylim(-.1,.1)
#plt.xlim(0,1)
"""
om1,V11,W11,eps1 = qs.unpack_arr(np.abs(inp.T[-1]),N)
om_beg,V_beg,W_beg,eps_beg = qs.unpack_arr(np.abs(inp.T[0]),N)

plt.rcParams.update({
    "text.usetex": True
})

#om = np.append(om1,om1[::-1])
#V = np.block([[V11,V11[:,::-1]],[V11[:,::-1].T,V11[::-1,::-1]]])
#W = np.block([[W11,W11[:,::-1]],[W11[:,::-1].T,W11[::-1,::-1]]])
#V11 = V[0:100,0:100]
#V22 = V[100:200,100:200]
#V12 = V[0:100,100:200]
#V21 = V[100:200,0:100]
#plt.show()


def visualize(arr1,arr2):
    arr1 = np.abs(arr1)
    arr2 = np.abs(arr2)
    vmax = max(np.append(arr1.flatten(),arr2.flatten()))
    
    fig, axes = plt.subplots(nrows=1, ncols=2)
    
    im1 = axes[0].imshow(arr1, aspect='auto', cmap='binary')
    im2 = axes[1].imshow(arr2, aspect='auto', cmap='binary')
    axes[0].set_xlabel(r"$\lambda = 0$")
    axes[1].set_xlabel(r"$\lambda = %.2f$"%t[-1])
    
    # add space for colour bar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im2, cax=cbar_ax)
    

import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.animation as animation


fps = 50
nSeconds = 20
snapshots = [ qs.unpack_arr(np.abs(inp.T[i]),N)[1] for i in range( nSeconds * fps ) ]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )

a = snapshots[0]
im = plt.imshow(a, norm=colors.LogNorm(vmin=1e-6, vmax=7),  aspect='auto', cmap='gray')
plt.colorbar(im, extend='max')

def animate_func(i):
    if i % fps == 0:
        print( '.', end ='' )

    im.set_array(snapshots[i])
    return [im]

anim = animation.FuncAnimation(
                               fig, 
                               animate_func, 
                               frames = nSeconds * fps,
                               interval = 1000 / fps, # in ms
                               )

anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

print('Done!')