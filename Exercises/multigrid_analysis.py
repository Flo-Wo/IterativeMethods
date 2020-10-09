# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 08:32:14 2020

@author: Michael Thiele
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator
from ReducedModel import fd_laplace, condition_number_factored, condition_number_normal,\
    multigrid_jacobi, get_system, multigrid_stat
from scipy import sparse
import matplotlib.pyplot as plt
plt.close('all')

# =============================================================================
# Test multigrid jacobi
# =============================================================================
 

def plot_mulitgrid_jacobi():
    omega = 0.5
    plt.figure(figsize = (20, 20))
    plt.suptitle(r"Using multigrid to solve the system $(\ nu \cdot A^2+  I )v = f$ with dampeed jacobi with damping parameter \omega = {}   ".format(omega))
    nu1 = 2
    nu2 = 2
    level = 3
    j=1
    for i in range(0,4):    
        nu= 0.01 * 10**i
        for l in range(4,7):
            
            m = 2**l -1
    
            A = fd_laplace(m,2)
            
            u_sol = np.random.rand(m**2)
            u_guess = np.zeros((m**2))
            
            A_2 = A.dot(A)
            S_jacobi = sparse.eye((m**2)) + nu * A_2        
            
            f =  S_jacobi.dot(u_sol)
            
            u_jac, res_jac, k_jac = multigrid_jacobi(nu, f, u_guess, m, omega, \
                                                     nu1, nu2, level)
            
            x_jac=np.arange(0, k_jac)
            
            plt.subplot(4,3,j)
            plt.semilogy(x_jac, res_jac, "r-", label="damped jacobi with {} levels".format(level))
            plt.legend(loc="upper right")
            plt.title(r"$m =$ {0}, $\nu =$ {1}".format(m, nu))
            print("norm differences = {}".format(np.linalg.norm(u_sol - u_jac)))
            j= j + 1
    #plt.tight_layout()
    plt.show()


# =============================================================================
# Test multigrid sationary method
# =============================================================================
def plot_mulitgrid_stat():
    plt.figure(figsize = (20, 20))
    plt.suptitle(r" Using multigrid to solve the system  $(\nu I + A^{-2} ) v = f $ with the given stationary method as smoother ")
    nu1 = 2
    nu2 = 2
    level = 3
    j=1
    for i in range(0,4):    
        nu= 100 * 10**i
        for l in range(4,7):
            
            m = 2**l -1

            u_sol = np.random.rand(m**2)
            u_guess = np.zeros((m**2))
            
            #construct linear operator
            
            op = get_system(m,nu)
            
            S_stat = LinearOperator((m**2,m**2),op)
            f_stat = S_stat(u_sol)
            
            u_stat, res_stat, k_stat = multigrid_stat(nu, f_stat, u_guess, m, \
                                                      nu1, nu2, level)
            
            x_stat = np.arange(0,k_stat)
            
            plt.subplot(4,3,j)

            plt.semilogy(x_stat, res_stat, "b-", label=" mutlti-grid  {} levels ".format(level))
            plt.legend(loc="upper right")
            plt.title(r"$m =$ {0}, $\nu =$ {1}".format(m, nu))
            print("norm differences = {}".format(np.linalg.norm(u_sol - u_stat)))
            j= j + 1
    #plt.tight_layout()
    plt.show()
    
plot_mulitgrid_jacobi()
plot_mulitgrid_stat()






















