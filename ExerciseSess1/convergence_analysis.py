#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:06:51 2020

@author: florianwolf
"""

import numpy as np
from ReducedModel import fd_laplace, solver_poisson, solver_poisson_factored
import matplotlib.pyplot as plt
plt.close('all')


def test_normal(nu_start, m, plot="nu"):
    y_d = 10*np.ones(m**2)
    
    u_sol = np.random.rand(m**2)
    
    u_guess = np.random.rand(m**2)
    
    A = fd_laplace(m,d=2)
    
    
    for nu in nu_start:
        f  = (-1)*(nu* A.dot(A.dot(u_sol)) + u_sol - A.dot(y_d))
        u_test,info,num_iters,res = solver_poisson(u_guess, nu, y_d, f,m)
        x = np.arange(0,num_iters)
        if plot == "nu":
            plt.semilogy(x,res, label="nu = {}".format(nu))
        else:
            plt.semilogy(x,res, label="m = {}".format(m))
        

def test_factored(nu_start, m, plot="nu"):
    y_d = 10*np.ones(m**2)
    
    u_sol = np.random.rand(m**2)
    
    u_guess = np.random.rand(m**2)
    
    A = fd_laplace(m,d=2)
    
    
    for nu in nu_start:
        f  = (-1)*(nu* A.dot(A.dot(u_sol)) + u_sol - A.dot(y_d))
        u_test,info,num_iters,res = solver_poisson_factored(u_guess, nu, y_d, f,m)
        x = np.arange(0,num_iters)
        if plot == "nu":
            plt.semilogy(x,res, label="nu = {}".format(nu))
        else:
            plt.semilogy(x,res, label="m = {}".format(m))


#------------------
# fixed m
#------------------

# ===============================
# Test normal system, fixed m
# ===============================
nu = [0.001 * 10**i for i in range(0,6)]
m = 32
step = 6
test_normal(nu, m, plot="nu")

plt.title("converence behaviour of the normal system, m = {1}".format(nu, m))
plt.legend(loc="upper right")
plt.xlabel("number of iterations")
plt.ylabel("Norm of the  residuals")
plt.show()

# ===============================
# Test factored system, fixed m
# ===============================


test_factored(nu, m, plot="nu")
    
plt.title("convergence behaviour of the factored system, m = {1}".format(nu, m))
plt.legend(loc="upper right")
plt.xlabel("number of iterations")
plt.ylabel("Norm of the  residuals")
plt.show()

#------------------
# fixed nu
#------------------

# ===============================
# Test normal system, fixed nu
# ===============================
nu = [0.01]
step = 1
for i in range(1, 6):
    test_normal(nu, 2**i, plot="m")

plt.title("converence behaviour of the normal system, nu = {0}".format(nu, m))
plt.legend(loc="upper right")
plt.xlabel("number of iterations")
plt.ylabel("Norm of the  residuals")
plt.show()

# ===============================
# Test factored system, fixed nu
# ===============================

for i in range(1, 6):
    test_factored(nu, 2**i, plot="m")
    
plt.title("convergence behaviour of the factored system, nu = {0}".format(nu, m))
plt.legend(loc="upper right")
plt.xlabel("number of iterations")
plt.ylabel("Norm of the  residuals")
plt.show()