# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 16:24:07 2023

@author: Jan-Philipp
"""

import numpy as np

bog_path = "C:\\Users\\Jan-Philipp\\Documents\\Eigene Dokumente\\Physikstudium\\6. Semester\\Bachelor-Thesis\\Numerics\\without_n_dep\\Julia\\with_offset_phi\\outfiles\\Bogoliubov,N=200,phi=0.1.csv"

etas = np.loadtxt(bog_path,usecols=(1),delimiter=',')