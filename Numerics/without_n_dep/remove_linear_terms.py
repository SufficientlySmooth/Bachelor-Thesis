# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 16:59:06 2023

@author: Jan-Philipp Christ
"""

import numpy as np
from scipy.optimize import fsolve

#-----------------constants----------------------
xi = 1 #characteristic length (BEC healing length)
cc = 1 #c in free Bogoliubov dispersion (speed of sound)
lamb_IR = 1e-1 #Infrared cutoff
lamb_UV = 1e1 #ultra-violet cutoff
m_b = 1/(np.sqrt(2)*cc*xi) #reduced mass = boson mass in the limit of infinite impurity mass
eta = 1 #will be varied between -10...10 later
n0 = 1.05/xi #
gamma = 0.438
a_bb = 2/(n0*gamma)
g_bb = -2/(m_b*a_bb)
g_ib = eta*g_bb
#------------------------------------------------

def om(k):
    """
    Parameters
    ----------
    k : wave numbers (float because we are in 1D)

    Returns
    -------
    float
         free Bogoliubov dispersion

    """
    return cc*k*np.sqrt(1+(k**2*xi**2)/2)

def W(k):
    """
    Parameters
    ----------
    k : wave numbers (float because we are in 1D)

    Returns
    -------
    float
        number which describes interaction strength.

    """
    return (k**2*xi**2/(2+k**2*xi**2))**(1/4)

def s(k):
    """
    Parameters
    ----------
    k : wave numbers (float because we are in 1D)

    Returns
    -------
    float
        interaction parameter in eq. 3 (see practical course manual)

    """
    w = W(k)
    return 1/2*(w+1/w)

def c(k):
    """
    Parameters
    ----------
    k : wave numbers (float because we are in 1D)

    Returns
    -------
    float
        interaction parameter in eq. 3 (see practical course manual)

    """
    w = W(k)
    return 1/2*(1/w-w)

def gen_1Dgrid(lamb_IR,lamb_UV,dk = 1e-1):
    """
    

    Parameters
    ----------
    lamb_IR : float
        IR cutoff.
    lamb_UV : float
        UV cutoff.
    dk : float
        spacing between k,k+1. The default is 1e-1.

    Raises
    ------
    ValueError
        If IR cutoff is not strictly smaller than the UV cutoff.

    Returns
    -------
    np.array
        contains all positive and negative values for k.

    """
    if lamb_UV<=lamb_IR:
        raise ValueError("IR cutoff has to be strictly smaller than the UV cutoff")
        
    n_min = int(np.floor(lamb_IR/dk))
    n_max = int(np.ceil(lamb_UV/dk))
    k_pos = np.array(list(range(n_min,n_max+1)))*dk
    k_neg = -k_pos[::-1]
    
    return np.append(k_neg,k_pos)

def V0(k,k_,L):
    """
    Parameters
    ----------
    k : wave number (float because we are in 1D)
    k_ : wave number (float because we are in 1D)
    L : 2pi/Delta k where Delta k is the spacing between k,k+1

    Returns
    -------
    float

    """
    return 2*np.pi*g_ib/L*(c(k)*c(k_)+s(k)*s(k_))
    
def W0(k,k_,L):
    """
    Parameters
    ----------
    k : wave number (float because we are in 1D)
    k_ : wave number (float because we are in 1D)
    L : 2pi/Delta k where Delta k is the spacing between k,k+1

    Returns
    -------
    float

    """
    return -2*np.pi*g_ib/L*s(k)*c(k_)

def eps0(grid):
    """
    Parameters
    ----------
    grid: np.array of wave numbers

    Returns
    -------
    float

    """
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk    
    return g_ib*n0+2*np.pi*g_ib/L*np.array([s(k)**2 for k in grid])

def omega0(k,L):
    """
    Parameters
    ----------
    k : wave number (float because we are in 1D)
    L : 2pi/Delta k where Delta k is the spacing between k,k+1

    Returns
    -------
    float

    """
    return 2*np.pi/L*om(k)

def W0_tilde(k,L):
    """
    Parameters
    ----------
    k : wave number (float because we are in 1D)
    L : 2pi/Delta k where Delta k is the spacing between k,k+1

    Returns
    -------
    float

    """
    return  g_ib*np.sqrt(n0)/np.sqrt(2*np.pi)*(2*np.pi/L)**(3/2)*W(k)

def omega0_arr(grid):
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk   
    return np.array([omega0(k,L) for k in grid])

def W0_tilde_arr(grid):
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk  
    return np.array([W0_tilde(k,L) for k in grid])

def W0_arr(grid):
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk  
    return np.array([[W0(k,k_,L) for k_ in grid] for k in grid])

def V0_arr(grid):
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk  
    return np.array([[V0(k,k_,L) for k_ in grid] for k in grid])
    
def func(alphas,grid):
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk  
    def alpha_(index): # returns alpha(-k)
        return alphas[len(grid)-(index+1)]
    return np.array([
        np.sum([V0(k,k_,L)*alphas[j]+W0(k_,k,L)*np.conj(alpha_(j))+W0(-k,k_,L)*np.conj(alphas[j]) for j in range(len(alphas)) if (k_:=grid[j])]) + W0_tilde(-k,L)
        for k in grid
        ])

def func2(alphas,grid):
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk  
    def alpha_(index): # returns alpha(-k)
        return alphas[len(grid)-(index+1)]
    return np.array([
        np.sum([V0(k_,k,L)*np.conj(alphas[j])+W0(-k,k_,L)*alpha_(j)+W0(k_,k,L)*alpha_(j) for j in range(len(alphas)) if (k_:=grid[j])]) + W0_tilde(k,L)
        for k in grid
        ])

def remove_linear(grid):  
    alpha0 = np.zeros(len(grid))
    sol = fsolve(func,x0=alpha0,args=(grid,))
    assert max(abs(func(sol,grid)))<1e-10 and max(abs(func2(sol,grid)))<1e-10, "Linear terms of Hamiltonian not successfully eliminated!"
    return sol

def get_quadratic_Hamiltonian(grid):
    dk = (grid[-1]-grid[-2])
    L = 2*np.pi/dk  
    alphas = remove_linear(grid)
    def alpha_(index): # returns alpha(-k)
        return alphas[len(grid)-(index+1)]
    eps_help_arr = np.array([[W0(k,k_,L)*(alpha_(i)*alphas[j]+np.conj(alpha_(i)*alphas[j]))+V0(k,k_,L)*np.conj(alphas[i]*alphas[j]) for i in range(len(grid)) if (k := grid[i])] for j in range(len(grid))  if (k_:=grid[j])])
    eps = eps0(grid) + np.sum([W0_tilde(k,L)*(alphas[i]+np.conj(alpha_(i))) for i in range(len(grid)) if (k:=grid[i])]) +np.sum(eps_help_arr.flatten())
    
    return omega0_arr(grid),V0_arr(grid),W0_arr(grid),eps

    

