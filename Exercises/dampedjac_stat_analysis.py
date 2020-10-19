# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:14:02 2020

@author: Thanh-Van Huynh
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from ReducedModel import fd_laplace, condition_number_factored, condition_number_normal,\
    multigrid_jacobi, get_system, multigrid_stat,solver_stationary_fixedRight,solver_damped_jacobi\
    ,laplace_eigs
from scipy import sparse
import matplotlib.pyplot as plt


def get_norm_jac_iteration_matrix(m,nu,omega):
    """
    Fuction to generate the norm of the iteration matrix solving the system
    
        (nu*A^2 + I)*u = A (y_d + A^{-1}*f) = A*y_d - f
        
    by using damped Jacobi

    Parameters
    ----------
    m : positive int
        size the matrix should have (for d=2 the matrix size is m**2 x m**2), 
        regarding your desired discretization
    nu : positive real number
        regularization parameter of the optimal control problem
    omega : real number between 0 and 1
        weighing parameter of the damped Jacobi iteration

    Returns
    -------
    Norm of the iteration matrix using damped Jacobi

    """
    A = fd_laplace(m,2).toarray()
    n = m**2
    A_2 = A.dot(A)
    System = np.eye(n)+ nu*A_2
    D=np.diag(np.diag(System))
    D_inv=omega *np.linalg.inv(D)
    It_matrix=D_inv.dot((1/omega)*D-System)
    return np.linalg.norm(It_matrix, ord=2)


def get_norm_stat_iteration_matrix(m,nu):
    """
    Fuction to generate the norm of the iteration matrix solving the system
    
        (nu*A^2 + I)*u = A (y_d + A^{-1}*f) = A*y_d - f
        
    by using the stationary method

    Parameters
    ----------
    m : positive int
        size the matrix should have (for d=2 the matrix size is m**2 x m**2), 
        regarding your desired discretization
    nu : positive real number
        regularization parameter of the optimal control problem

    Returns
    -------
    Norm of the iteration matrix using the stationary method

    """
    lam,_ = laplace_eigs(m)
    lam = lam**2
    lam=np.abs(lam)
    # smallest ev of A^2 
    min_ev= np.min(lam)
    #bigest ev of A^-2 = ||A^-2||
    max_ev= (1/ min_ev)
    norm= (1/nu )* max_ev 
    return norm

def plot_damped_jac(omega,m,nu):
    """
    Function to analyse the convergence of the solving the system
    
        (nu*A^2 + I)*u = A (y_d + A^{-1}*f) = A*y_d - f
        
    by using damped Jacobi

    Parameters
    ----------
    m : positive int
        size the matrix should have (for d=2 the matrix size is m**2 x m**2), 
        regarding your desired discretization
    nu : positive real number
        regularization parameter of the optimal control problem
    omega : real number between 0 and 1
        weighing parameter of the damped Jacobi iteration
    option : 1=omega, 2=nu, 3=m
        choosing which parameter to modify

    Returns
    -------
    Three subplots with different omega, m or nu
    
    """    
    plt.figure(figsize = (15, 15))
    
    #counter for subplots
    j = 1
    #create test vectors
    y_d = 10*np.ones(m**2)
    u_guess = np.random.rand(m**2)
    f = np.random.rand(m**2)
    
    #fixed m and nu, modifying omega
    for i in range(0,3):
        #using damped Jacobi
        omega = 0.25*(i+1)
        u,k,res = solver_damped_jacobi_mod(omega, nu, m, y_d, f, u_guess, maxIter=1000,tol=1e-8, norm=())
        #plotting
        it = np.arange(0,k,1)
        plt.subplot(3,3,j)
        plt.semilogy(res,it, "r-",label="$\omega$={}".format(omega))
        plt.legend(loc="upper right")
        plt.ylabel("norm of residual")
        plt.xlabel("number of iterations")
        plt.title(r"$m =$ {0}, $\nu =$ {1}".format(m, nu))
        j += 1
    
    #fixed omega and m, modifying nu      
    for i in range(0,3):
        #using damped Jacobi
        nu = 0.1**(i+1)
        u,k,res = solver_damped_jacobi_mod(omega, nu, m, y_d, f, u_guess, maxIter=1000,tol=1e-8, norm=())
        #plotting
        it = np.arange(0,k,1)
        plt.subplot(3,3,j)
        plt.semilogy(res,it, "r-",label="$\nu$={}".format(nu))
        plt.legend(loc="upper right")
        plt.ylabel("norm of residual")
        plt.xlabel("number of iterations")
        plt.title(r"$\omega =$ {0}, m= {1}".format(omega,m))
        j += 1
    
    #fixed omega and nu, modifying m
    for l in range(4,7):
        m = 2**l-1
        #creating test vectors and using damped Jacobi
        y_d = 10*np.ones(m**2)
        u_guess = np.random.rand(m**2)
        f = np.random.rand(m**2)
        u,k,res = solver_damped_jacobi_mod(omega, nu, m, y_d, f, u_guess, maxIter=1000,tol=1e-8, norm=())
        #plotting
        it = np.arange(0,k,1)
        plt.subplot(3,3,j)
        plt.semilogy(res,it, "r-",label="m={}".format(m))
        plt.legend(loc="upper right")
        plt.ylabel("norm of residual")
        plt.xlabel("number of iterations")
        plt.title(r"$\omega =$ {0}, $\nu =$ {1}".format(omega, nu))
        j += 1
        