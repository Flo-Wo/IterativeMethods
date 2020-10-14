# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:06:59 2020

@author: Michael Thiele
"""

import numpy as np
from scipy.sparse.linalg import LinearOperator
from ReducedModel import fd_laplace, condition_number_factored, condition_number_normal,\
    multigrid_jacobi, get_system, multigrid_stat,solver_stationary_fixedRight,solver_damped_jacobi\
    ,laplace_eigs
from scipy import sparse
import matplotlib.pyplot as plt


def get_norm_jac_iteration_matrix(m,nu,omega):
    A = fd_laplace(m,2).toarray()
    n=m**2
    A_2=A.dot(A)
    System= np.eye(n)+ nu*A_2
    D=np.diag(np.diag(System))
    D_inv=omega *np.linalg.inv(D)
    It_matrix=D_inv.dot(System)
    return np.linalg.norm(It_matrix)


def get_norm_stat_iteration_matrix(m,nu):
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
    
    for s in range(0,4):
        omega=0.125 * 2**s
        for l in range(4,7):
                        
            m = 2**l -1
            it=7
            y=np.zeros((it,1))
            nu=np.zeros((it,1))
            for i in range(0,it):
                nu[i] = 0.00001*10**i
                y[i]=get_norm_jac_iteration_matrix(m,nu[i],omega)
            
            x=nu
            plt.subplot(4,3,j)
            
            plt.semilogy(x, y, "r-", label="norm of iteration matrix of jacobi")
            plt.legend(loc="upper right")
            plt.title(r"$m =$ {0},   $\omega=$ {1}".format(m, omega))
            plt.ylabel("norm ")
            j= j + 1
    #plt.tight_layout()
    plt.show()
    
def plot_norm_iteration_matrix_stat():
    plt.figure(figsize = (15, 15))
    plt.suptitle(r" norm of iteration matrix of the stationary method which is $|| \nu A^{-2} || $")
    j=1
    
    for l in range(4,7):
                    
        m = 2**l -1
        it=7
        y=np.zeros((it,1))
        nu=np.zeros((it,1))
        for i in range(0,it):
            nu[i] = 0.00001*10**i
            y[i]=get_norm_stat_iteration_matrix(m,nu[i])
        
        x=nu
        plt.subplot(1,3,j)
        
        plt.semilogy(x, y, "r-", label="norm of iteration matrix of the stationary method")
        plt.legend(loc="upper right")
        plt.title(r"$m =$ {0} ".format(m))
        plt.ylabel("norm ")
        j= j + 1
    #plt.tight_layout()
    plt.show()
    
def plot_norm_iteration_matrix_stat_jac():
    plt.figure(figsize = (15, 15))
    j=1
    
    for s in range(0,4):
        omega=0.125 * 2**s
        for l in range(4,7):
                        
            m = 2**l -1
            it=7
            y_jac=np.zeros((it,1))
            y_stat=np.zeros((it,1))
            nu=np.zeros((it,1))
            for i in range(0,it):
                nu[i] = 0.00001*10**i
                y_stat[i]=get_norm_stat_iteration_matrix(m,nu[i])
                y_jac[i]=get_norm_jac_iteration_matrix(m,nu[i],omega)
            
            x=nu
            plt.subplot(4,3,j)
            
            plt.semilogy(x, y_stat, "r-", label="norm of iteration matrix of stationary method")
            plt.semilogy(x, y_jac, "b-", label="norm of iteration matrix of damped")
            plt.legend(loc="upper right")
            plt.title(r"$m =$ {0},   $\omega=$ {1}".format(m, omega))
            plt.ylabel("norm ")
            j= j + 1
    #plt.tight_layout()
    plt.show()

n=256
m=16
nu=1000
omega=0.3
plot_norm_iteration_matrix_jac()
plot_norm_iteration_matrix_stat()
plot_norm_iteration_matrix_stat_jac()