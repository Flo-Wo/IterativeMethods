#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 19:30:23 2020

@author: florianwolf
"""

import inspect
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from ReducedModel import fd_laplace, solver_poisson_unfactored_cg, solver_poisson_factored_cg,\
    solver_stationary, solver_damped_jacobi, condition_number_factored, condition_number_normal,\
    multigrid_jacobi, get_system, multigrid_stat
import matplotlib.pyplot as plt
plt.close('all')




def test_normal(nu_start, m, plot="nu"):
    y_d = 10*np.ones(m**2)
    
    u_sol = np.random.rand(m**2)
    
    u_guess = np.random.rand(m**2)
    
    A = fd_laplace(m,d=2)
    
    
    for nu in nu_start:
        f  = (-1)*(nu* A.dot(A.dot(u_sol)) + u_sol - A.dot(y_d))
        u_test,info,num_iters,res = solver_poisson_unfactored_cg(u_guess, nu, y_d, f, m)
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
        u_test,info,num_iters,res = solver_poisson_factored_cg(u_guess, nu, y_d, f, m)
        x = np.arange(0,num_iters)
        if plot == "nu":
            plt.semilogy(x,res, label="nu = {}".format(nu))
        else:
            plt.semilogy(x,res, label="m = {}".format(m))


plt.figure(figsize = (30, 30))
j = 1
for k in range(4):
    nu = 0.01 * 10**k
    for l in range(4,7):
        # create test vectors to get the system behaviour
        m = 2**l-1
        y_d = 10*np.ones(m**2)
        
        u_sol = np.random.rand(m**2)
        
        u_guess = np.random.rand(m**2)
        
        A = fd_laplace(m,d=2)

        # calculate f, as the real solutions is already known
        f  = (-1)*(nu* A.dot(A.dot(u_sol)) + u_sol - A.dot(y_d))
        # use the solver of the unfactored system
        u_test,info ,num_iters ,res = solver_poisson_unfactored_cg(u_guess, nu, y_d, f, m)
        u_test_factored,info_factored,num_iters_factored,res_factored = \
            solver_poisson_factored_cg(u_guess, nu, y_d, f, m)
        x = np.arange(0, num_iters)
        x_factored = np.arange(0, num_iters_factored)
        plt.subplot(4,3,j)
        plt.semilogy(x, res, "r-", label="initial system")
        plt.semilogy(x_factored, res_factored, "b-", label="factored system")
        plt.legend(loc="upper right")
        plt.title(r"$m =$ {0}, $\nu =$ {1}".format(m, nu))
        j+=1
plt.tight_layout()
plt.show()
        
        

    
# #------------------
# # fixed m
# #------------------

# # ===============================
# # Test normal system, fixed m
# # ===============================
# nu = [0.01 * 10**i for i in range(0,4)]
# m = 31
# step = 6
# test_normal(nu, m, plot="nu")

# plt.title("convergence behaviour of the normal system (cg), m = {1}".format(nu, m))
# plt.legend(loc="upper right")
# plt.xlabel("number of iterations")
# plt.ylabel("Norm of the  residuals")
# plt.show()

# # ===============================
# # Test factored system, fixed m
# # ===============================


# test_factored(nu, m, plot="nu")
    
# plt.title("convergence behaviour of the factored system (cg), m = {1}".format(nu, m))
# plt.legend(loc="upper right")
# plt.xlabel("number of iterations")
# plt.ylabel("Norm of the  residuals")
# plt.show()

# #------------------
# # fixed nu
# #------------------

# # ===============================
# # Test normal system, fixed nu
# # ===============================
# nu = [0.1]
# step = 1
# for i in range(4, 7):
#     test_normal(nu, 2**i-1, plot="m")

# plt.title("convergence behaviour of the normal system (cg), nu = {0}".format(nu, m))
# plt.legend(loc="upper right")
# plt.xlabel("number of iterations")
# plt.ylabel("Norm of the  residuals")
# plt.show()

# # ===============================
# # Test factored system, fixed nu
# # ===============================
# nu = [10]
# for i in range(4, 7):
#     test_factored(nu, 2**i-1, plot="m")
    
# plt.title("convergence behaviour of the factored system (cg), nu = {0}".format(nu, m))
# plt.legend(loc="upper right")
# plt.xlabel("number of iterations")
# plt.ylabel("Norm of the  residuals")
# plt.show()