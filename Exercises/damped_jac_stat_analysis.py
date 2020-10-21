# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:06:59 2020

@author: Michael Thiele
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from ReducedModel import fd_laplace, condition_number_factored, condition_number_normal,\
    multigrid_jacobi, get_system, multigrid_stat,solver_stationary_fixedRight,solver_damped_jacobi\
    ,laplace_eigs, solver_damped_jacobi_mod
from scipy import sparse
import matplotlib.pyplot as plt
plt.close('all')


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
        danping parameter of the damped Jacobi iteration

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

def plot_norm_iteration_matrix_jac():
    plt.figure(figsize = (15, 15))
    plt.suptitle(r" norm of iteration matrix of damped jacobi, ehich is $|| \omega D^{-1} (I+ \nu A^2) || $, where D=diag( I+ \nu A^2)")
    j=1
    
    for s in range(1,5):
        omega=0.125*2**s
        it = 10
        for l in range(4,6):
                        
            m = 2**l -1
            y=np.zeros(it)
            nu=np.zeros(it)
            for i in range(0,it):
                nu[i] = 0.00001*10**i
                y[i]=get_norm_jac_iteration_matrix(m,nu[i],omega)
            
            x=nu
            plt.subplot(4,3,j)
            
            plt.semilogx(x, y, "r-", label=r"$\omega$ = {}".format(omega))
            plt.legend(loc="upper right")
            plt.title(r"$m =$ {0},   $\omega=$ {1}".format(m, omega))
            plt.ylabel("norm ")
            plt.xticks(x)
            plt.yscale("log")
            j= j + 1
    plt.tight_layout()
    plt.show()
    
def plot_norm_iteration_matrix_stat():
    plt.figure(figsize = (15, 15))
    plt.suptitle(r" norm of iteration matrix of the stationary method which is $|| \nu A^{-2} || $")
    j=1
    
    for l in range(4,5):
                    
        m = 2**l -1
        it=7
        y=np.zeros(it)
        nu=np.zeros(it)
        for i in range(0,it):
            nu[i] = 0.00001*10**i
            y[i]=get_norm_stat_iteration_matrix(m,nu[i])
        
        x=nu
        plt.subplot(1,3,j)
        
        plt.semilogx(x, y, "r-", label="norm of iteration matrix of the stationary method")
        plt.legend(loc="upper right")
        plt.title(r"$m =$ {0} ".format(m))
        plt.ylabel("norm ")
        j= j + 1
    #plt.tight_layout()
    plt.show()
    
def plot_norm_iteration_matrix_stat_jac():
     plt.figure(figsize = (15, 15))
     j=1
    
     for s in range(0,1):
         omega=0.125 * 2**s
         for l in range(4,5):
                        
             m = 2**l -1
             it=12
             y_jac=np.zeros(it)
             y_stat=np.zeros(it)
             nu=np.zeros((it,1))
             for i in range(0,it):
                 nu[i] = 0.00001*10**i
                 y_stat[i]=get_norm_stat_iteration_matrix(m,nu[i])
                 y_jac[i]=get_norm_jac_iteration_matrix(m,nu[i],omega)
            
             x=nu
             plt.subplot(4,3,j)
            
             plt.semilogx(x, y_stat, "r-", label="stationary method")
             plt.semilogx(x, y_jac, "b-", label="damped")
             plt.legend(loc="upper right")
             plt.title(r"$m =$ {0},   $\omega=$ {1}".format(m, omega))
             plt.ylabel("norm ")
             plt.yscale("log")
             j= j + 1
     plt.tight_layout()
     plt.show()

def plot_damped_jac(omega,m,nu,option):
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
    Three subplots with different omega, nu or m
    
    """    
    plt.figure(figsize = (20, 20))
    maxIter = 1000
    tol = 1e-8
    
    #fixed m and nu, modifying omega
    if option == 1:
        #counter for subplots
        j = 1
        #create test vectors
        y_d = 10*np.ones(m**2)
        u_guess = np.random.rand(m**2)
        f = np.random.rand(m**2)
        
        for i in range(0,3):
            #using damped Jacobi
            omega = 0.3*(i+1)
            u,k,res = solver_damped_jacobi_mod(omega, nu, m, y_d, f, u_guess, maxIter,tol, norm=())
            #plotting
            it = np.arange(0,k,1)
            plt.subplot(3,3,j)
            plt.semilogy(it,res, "r-",label="$\omega$={}".format(omega))
            plt.legend(loc="upper right")
            plt.ylabel("norm of residual")
            plt.xlabel("number of iterations")
            plt.title(r"$m =$ {0}, $\nu =$ {1}".format(m, nu))
            j += 1
    
    #fixed omega and m, modifying nu
    if option == 2:
        #counter for subplots
        j = 4
        #create test vectors
        y_d = 10*np.ones(m**2)
        u_guess = np.random.rand(m**2)
        f = np.random.rand(m**2)
        
        for i in range(0,3):
            #using damped Jacobi
            nu = 10**(-(3*i+1))
            u,k,res = solver_damped_jacobi_mod(omega, nu, m, y_d, f, u_guess, maxIter,tol, norm=())
            #plotting
            it = np.arange(0,k,1)
            plt.subplot(3,3,j)
            #plt.semilogy(res,it, "r-",label="$\nu =${}".format(nu)) irgendwie zeigt Python mir hier das nu nicht gescheit an in der Legende...
            plt.semilogy(it,res, "r-",label="nu ={}".format(nu))
            plt.legend(loc="upper right")
            plt.ylabel("norm of residual")
            plt.xlabel("number of iterations")
            plt.title(r"$\omega =$ {0}, m= {1}".format(omega,m))
            j += 1
    
    #fixed omega and nu, modifying m
    if option == 3:
        j = 7
        for l in range(4,7):
            m = 2**l-1
            #creating test vectors and using damped Jacobi
            y_d = 10*np.ones(m**2)
            u_guess = np.random.rand(m**2)
            f = np.random.rand(m**2)
            u,k,res = solver_damped_jacobi_mod(omega, nu, m, y_d, f, u_guess, maxIter,tol, norm=())
            #plotting
            it = np.arange(0,k,1)
            plt.subplot(3,3,j)
            plt.semilogy(it,res, "r-",label="m={}".format(m))
            plt.legend(loc="upper right")
            plt.ylabel("norm of residual")
            plt.xlabel("number of iterations")
            plt.title(r"$\omega =$ {0}, $\nu =$ {1}".format(omega, nu))
            j += 1

#test
for i in range(1,4):
    plot_damped_jac(omega=0.6,m=15,nu=0.00001,option=i)

plot_norm_iteration_matrix_jac()
plot_norm_iteration_matrix_stat()
#plot_norm_iteration_matrix_stat_jac()