# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:25:37 2023

@author: Jan-Philipp
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.style.use('JaPh')


PATH = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelorarbeit_sol_files\\N=40,different etas_full_with_phi,phi=0.1\\"#N=40,lambda_IR=0.1+phi,lmabda_UV=10+phi\\"

N = 40


def unpack_arr(flat,N):
    N = int(N)
    om0 = flat[0:N]
    V0 = flat[N:N+int(N**2)]
    W0 = flat[N+int(N**2):N+int(2*N**2)]
    eps = flat[-1]
    V = V0.reshape((N,N))
    W = W0.reshape((N,N))
    return om0,V,W,eps

fig,(ax1,ax2)=plt.subplots(2,1,figsize=(10,10/np.sqrt(2)),sharex=True) 

ax3 = fig.add_subplot(111, zorder=-1)
#ax3.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

for _, spine in ax3.spines.items():
   spine.set_visible(False)
#ax1.grid()
#ax2.grid()


ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False,length=0)
ax3.set_xlim(0.00000001,30)
#
ax3.get_shared_x_axes().join(ax3, ax1)
ax3.set_xscale('symlog')

#ax2.set_xscale('log')
#ax3.grid(axis="x")
ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False,length=0)
ax1.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
ax2.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
#ax3.set_xticks([])
ax3.set_yticks([])
#print(ax3.xaxis.get_minor_locator())
#ax3.set_xticks([0,1,10])

"""
for line in ax3.get_xminorgridlines():
    print(line.set_color('red'))
"""

#ax1.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
#ax2.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
for FILENAME in [file for file in os.listdir(PATH) if not "_t" in file]:
    eta = float(FILENAME.split(',')[0].split('=')[-1])
    t_filename = "sol_full_t"+FILENAME[13:]
    if eta == 3:
        path_inp = PATH+FILENAME
        path_t = PATH+t_filename
        
        inp = np.load(path_inp)
        
        t = np.load(path_t)
        omegas = inp.T[:,0:N]
    if eta == -3:
        path_inp = PATH+FILENAME
        path_t = PATH+t_filename
        
        inp = np.load(path_inp)
        
        t = np.load(path_t)
        omegas2 = inp.T[:,0:N]
        


ax1.tick_params(axis='both',labelsize=10)
#ax1.set_xlabel(r'$\lambda[\xi^2/c^2]$',fontsize = 16)
#ax1.set_ylabel(r'Eigenenergies $\omega[c/\xi]$',fontsize = 16)
ax1.plot(t,omegas[:,0],marker='None',linestyle='-',linewidth=.7,color='firebrick',label=r'$\eta = 3$')
ax1.plot(t,omegas2[:,0],marker='None',linestyle='-',linewidth=.7,color='steelblue',label=r'$\eta = -3$')
ax1.plot(t,omegas,marker='None',linestyle='-',linewidth=.7,color='firebrick')
ax1.plot(t,omegas2,marker='None',linestyle='-',linewidth=.7,color='steelblue')
#ax1.set_yscale('linear')
#ax1.set_xscale('linear')
ax1.set_xlim(0,max(t))
ax1.set_ylim(64.01,72.5)
ax1.set_xscale('log')
ax2.set_xscale('log')
ax1.legend(loc='best',fontsize=12)

ax2.tick_params(axis='both',labelsize=10)
ax2.set_xlabel(r'$\lambda[\xi^2/c^2]$',fontsize = 16)
fig.supylabel(r'Eigenenergies $\omega[c/\xi]$',fontsize = 16)
ax2.plot(t,omegas,marker='None',linestyle='-',linewidth=.7,color='firebrick')
ax2.plot(t,omegas2,marker='None',linestyle='-',linewidth=.7,color='steelblue')
#ax2.set_yscale('linear')
#ax2.set_xscale('linear')
ax2.set_xlim(0,max(t))
ax2.set_ylim(0,3.99)
#ax2.set_xscale('log')
#ax1.set_xscale('log')

ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False,marker='None')
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

#ax1.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
#ax2.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

#ygridlines = ax3.get_xticklines()

#gridline_of_interest = ygridlines[0]
#gridline_of_interest.set_visible(False)

#ax1.legend(loc='best',fontsize=14,ncols=2)
for xmin in ax3.xaxis.get_minorticklocs():
  if not xmin == 30:
      ax3.axvline(x=xmin, linestyle='-',linewidth=.008, alpha=.2,color='grey')

plt.tight_layout()
fig.subplots_adjust(wspace=0, hspace=.03)
plt.savefig('Convergence_analysis,N=40.pdf',dpi=300)
plt.show()


"""
ax3 = fig.add_subplot(111, zorder=-1)


for _, spine in ax3.spines.items():
   spine.set_visible(False)
   
#ax1.grid()
#ax2.grid()

ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False,length=0)
ax3.set_xlim(0.00000001,30)
#
ax3.get_shared_x_axes().join(ax3, ax1)
ax3.set_xscale('symlog')

#ax2.set_xscale('log')
#ax3.grid(axis="x")
ax3.tick_params(labelleft=False, labelbottom=False, left=False, right=False,length=0)
#ax3.set_xticks([])
ax3.set_yticks([])
"""