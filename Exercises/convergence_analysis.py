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

def plot_stationary(m,nu,option):
    """
    Function to analyse the convergence of the initial optimality system 
    
        (nu * A^-2)u = A^-1 * y_d - A^-2 * f
    
    using the iteration process
    
        u_{n+1} = -1/nu A^{-2} u_n + 1/nu (A^-1*(y_d- A^-1 *f))

    Parameters
    ----------
    m : positive int
        size the matrix should have (for d=2 the matrix size is m**2 x m**2), 
        regarding your desired discretization
    nu : positive real number
        regularization parameter of the optimal control problem
    option : 1=nu, 2=m
        choosing which parameter to modify

    Returns
    -------
    Three subplots with different nu or m
    
    """    
    plt.figure(figsize = (20, 20))
    maxIter = 1000
    tol = 1e-8
    
    #fixed m, modifying n
    if option == 1:
        #counter for subplots
        j = 1
        
        for i in range(0,3):
            #create test vectors
            y_d = 10*np.ones(m**2)
            u_guess = np.random.rand(m**2)
            f = np.random.rand(m**2)
            #using stationary method
            nu = 10**(-(i+1))
            u,k,res = solver_stationary(u_guess,nu, y_d, f, m, maxIter=1000, tol=1e-8)
            #plotting
            it = np.arange(0,k,1)
            plt.subplot(1,3,j)
            plt.semilogy(it,res, "r-",label="nu={}".format(nu))
            plt.legend(loc="upper right")
            plt.ylabel("norm of residual")
            plt.xlabel("number of iterations")
            plt.title(r"$m =$ {0}".format(m))
            j += 1
    
    #fixed nu, modifying m
    if option == 2:
        #counter for subplots
        j = 1
        
        for l in range(3,6):
            m = 2**l-1
            #create test vectors
            y_d = 10*np.ones(m**2)
            u_guess = np.random.rand(m**2)
            f = np.random.rand(m**2)
            #using stationary method
            u,k,res = solver_stationary(u_guess,nu, y_d, f, m, maxIter=1000, tol=1e-8)
            #plotting
            it = np.arange(0,k,1)
            plt.subplot(1,3,j)
            plt.semilogy(it,res, "r-",label="m={}".format(m))
            plt.legend(loc="upper right")
            plt.ylabel("norm of residual")
            plt.xlabel("number of iterations")
            plt.title(r"$\nu =$ {0}".format(nu))
            j += 1

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
            
            
# # ===============================
# # Test initial system using stationary method, fixed m
# # ===============================
plot_stationary(m=15,nu=0.01,option=1)

# # ===============================
# # Test initial system using stationary method, fixed nu
# # ===============================
plot_stationary(m=15,nu=0.01,option=2)