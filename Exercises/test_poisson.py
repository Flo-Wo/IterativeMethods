#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 14:51:02 2020

@author: florianwolf
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from ReducedModel import fd_laplace, condition_number_factored, condition_number_normal,\
    multigrid_jacobi, get_system, multigrid_stat,solver_stationary_fixedRight,solver_damped_jacobi,\
    fast_poisson, laplace_small_decomposition
    
from scipy import sparse
import matplotlib.pyplot as plt
import timeit
plt.close('all')

l = 6

m = 2**l -1

A = fd_laplace(m, d=2)

for i in range(100):
    x_sol = np.random.rand(m**2)
    y = A.dot(x_sol)
    
    lam, V = laplace_small_decomposition(m)
    
    x_test = fast_poisson(V, V, lam, lam, y)
    
    print("norm of the differences = {}".format(np.linalg.norm(x_sol - x_test)))