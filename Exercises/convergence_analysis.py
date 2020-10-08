#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 16:06:51 2020

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
        u_test,info,num_iters,res = solver_poisson_unfactored_cg(u_guess, nu, y_d, f,m)
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
        u_test,info,num_iters,res = solver_poisson_factored_cg(u_guess, nu, y_d, f,m)
        x = np.arange(0,num_iters)
        if plot == "nu":
            plt.semilogy(x,res, label="nu = {}".format(nu))
        else:
            plt.semilogy(x,res, label="m = {}".format(m))

def test_stationary(nu_start, m, plot="nu", toler=1e-12):
    y_d = 10*np.ones(m**2)
    
    u_sol = np.random.rand(m**2)
    
    u_guess = np.random.rand(m**2)
    
    A = fd_laplace(m,d=2)
    
    
    for nu in nu_start:
        f  = (-1)*(nu* A.dot(A.dot(u_sol)) + u_sol - A.dot(y_d))
        u_test,num_iters,res = solver_stationary(u_guess, nu, y_d, f,m, tol=toler)
        x = np.arange(0,num_iters)
        if plot == "nu":
            plt.semilogy(x,res, label="nu = {}".format(nu))
        else:
            plt.semilogy(x,res, label="m = {}".format(m))
            
            
def test_jacobi(nu_start, m, omega, plot="nu"):
    #y_d = 10*np.random.rand(m**2)
    
    #f=np.random.rand(m**2) 
    u_sol = np.random.rand(m**2)
    u_guess = np.random.rand(m**2)
    
    

    maxIter=10000
    
    for nu in nu_start:
        A = fd_laplace(m,2)
        A_2 = np.dot(A,A)
        C = sparse.eye((m**2)) + nu * A_2
        
        f = C.dot(u_sol)
        
        #f  = A.dot(y_d) - f
        u_test, num_iters, res = solver_damped_jacobi(C, u_guess, omega, f, maxIter)
        x = np.arange(0,num_iters)
        
        #u_test2 = sparse.linalg.spsolve(C, f)
        
        #print("norm error = {}".format(np.linalg.norm(u_sol - u_test)))
        if plot == "nu":
            plt.semilogy(x,res, label="nu = {}".format(nu))
        else:
            plt.semilogy(x,res, label="m = {}".format(m))


# # # #### Test multigrid jacobi

# l = 4

# m = 2**l -1
# nu = 0.5

# A = fd_laplace(m,2)

# u_sol = np.random.rand(m**2)
# u_guess = np.random.rand(m**2)

# A_2 = A.dot(A)
# C = sparse.eye((m**2)) + nu * A_2        
# f = C.dot(u_sol)
# omega = 0.5

# nu1 = 2
# nu2 = 2
# level = 4

# u_res, res, k = multigrid_jacobi(nu, f, u_guess, m, omega, nu1, nu2, level)

# print("norm differences = {}".format(np.linalg.norm(u_sol - u_res)))


#### Test multigrid stationary

l = 4

m = 2**l -1
nu = 1e3

  #construct linear operator
op = get_system(m,nu)
A = LinearOperator((m**2,m**2),op)

u_sol = np.random.rand(m**2)
u_guess = np.random.rand(m**2)
      
f = A(u_sol)

nu1 = 2
nu2 = 2
level = 4

u_res, res, k = multigrid_stat(nu, f, u_guess, m, nu1, nu2, level)

print("norm differences = {}".format(np.linalg.norm(u_sol - u_res)))

# #------------------
# # fixed m
# #------------------

# # ===============================
# # Test normal system, fixed m
# # ===============================
# nu = [0.001 * 10**i for i in range(0,6)]
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
# nu = [0.01]
# step = 1
# for i in range(2, 6):
#     test_normal(nu, 2**i-1, plot="m")

# plt.title("convergence behaviour of the normal system (cg), nu = {0}".format(nu, m))
# plt.legend(loc="upper right")
# plt.xlabel("number of iterations")
# plt.ylabel("Norm of the  residuals")
# plt.show()

# # ===============================
# # Test factored system, fixed nu
# # ===============================

# for i in range(2, 6):
#     test_factored(nu, 2**i-1, plot="m")
    
# plt.title("convergence behaviour of the factored system (cg), nu = {0}".format(nu, m))
# plt.legend(loc="upper right")
# plt.xlabel("number of iterations")
# plt.ylabel("Norm of the  residuals")
# plt.show()
            
            
# #####
# # fixed nu              
# #####


# omega=0.5
# nu = [0.01 * 10**i for i in range(0,5)]
# #nu = [1]
# m = 8
# step = 6


# test_jacobi(nu, m, omega, plot="nu")
    
# plt.title("convergence behaviour damped jacobi factored system, m = {1} and omega = {2}".format(nu, m,omega))
# plt.legend(loc="upper right")
# plt.xlabel("number of iterations")
# plt.ylabel("Norm of the  residuals")
# plt.show()


# nu = [0.1 * 10**i for i in range(0,10)]
# m = 4

# cond = []
# cond_compare = []

# for nu_cur in nu:
#     cond.append(condition_number_factored(m, nu_cur))
#     A = fd_laplace(m, 2)
#     matrix = nu_cur*A.dot(A) + sparse.eye(m**2)
#     cond_compare.append(np.linalg.cond(matrix.toarray()))
    
# x = np.arange(0,10)
# plt.plot(x, cond, "rx-")
# plt.plot(x, cond_compare, "bx-")
            
            
