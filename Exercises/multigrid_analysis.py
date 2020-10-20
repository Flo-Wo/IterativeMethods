# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 08:32:14 2020

@author: Michael Thiele
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

# =============================================================================
# Test multigrid jacobi
# =============================================================================

def plot_mulitgrid_jacobi():
    omega = 2/3
    plt.figure(figsize = (20, 20))
    #plt.suptitle(r"Using multigrid to solve the system $(\ nu \cdot A^2+  I )v = f$ with dampeed jacobi with damping parameter $\omega =$ {}   and  comparing this to the convergence of the smoother as a stationary method".format(omega))
    nu1 = 3
    nu2 = 3
    level = 3
    j=1
    for i in range(0,4):    
        nu= 0.01 * 10**i
        for l in range(4,7):
            
            m = 2**l -1
            h = 1/(m+1)
            A = fd_laplace(m,2)
            
            u_sol = np.random.rand(m**2)
            u_guess = np.zeros(m**2)
            
            A_2 = A.dot(A)
            S_jacobi = sparse.eye((m**2)) + nu * A_2        
            
            f =  S_jacobi.dot(u_sol)
            
            u_jac, res_jac, k_jac = multigrid_jacobi(nu, f, u_guess, m, omega, \
                                                     nu1, nu2, level)
            (u_jac_m, k_jac_m, res_jac_m)= solver_damped_jacobi(A, u_guess, omega, f, maxIter=k_jac,tol=1e-6)
            
            x_jac=np.arange(0, k_jac)
            x_jac_m=np.arange(0, k_jac_m)
            
            plt.subplot(4,3,j)
            plt.semilogy(x_jac, res_jac, "b-", label="multigrid with {} levels".format(level))
            plt.semilogy(x_jac_m, res_jac_m, "r-", label="damped jacobi")
            plt.legend(loc="upper right")
            plt.ylabel("norm of residual ")
            plt.xlabel("iterations")
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
    #plt.suptitle(r" Using multigrid to solve the system  $(\nu I + A^{-2} ) v = f $ with the given stationary method as smoother and  comparing this to the convergence of the smoother as a stationary method")
    nu1 = 2
    nu2 = 2
    level = 3
    j=1
    for i in range(0,4):    
        nu= 0.01 * 10**i
        for l in range(4,7):
            
            m = 2**l -1
            
            u_sol = np.zeros(m**2)
            u_guess = np.zeros(m**2)
            
            #construct linear operator
            
            f = - np.ones(m**2)
            y_d = np.ones(m**2)
            
            lam, V = laplace_small_decomposition(m)
            f_temp = (-1)*(y_d + fast_poisson(V, V, lam, lam, f))
            
            f_stat = fast_poisson(V, V, lam, lam, f_temp)
            #print(f_stat)
            #print("\n\n")
            
            # op = get_system(m,nu)
            
            # S_stat = LinearOperator((m**2,m**2),op)
            # f_stat = S_stat(u_sol)
            
            u_stat, res_stat, k_stat = multigrid_stat(nu, f_stat, u_guess, m, \
                                                      nu1, nu2, level)
            print(res_stat)
            u_stat_m, k_stat_m, res_stat_m = solver_stationary_fixedRight(u_guess,nu,\
                                                                          f_stat, m, maxIter=k_stat,tol=1e-6)
            x_stat = np.arange(0,k_stat)
            x_stat_m = np.arange(0,k_stat_m)
            
            plt.subplot(4,3,j)
            
            plt.semilogy(x_stat_m, res_stat_m, "r-", label="stationary method ")
            plt.semilogy(x_stat, res_stat, "b-", label=" mutlti-grid  {} levels ".format(level))
            plt.legend(loc="upper right")
            plt.title(r"$m =$ {0}, $\nu =$ {1}".format(m, nu))
            plt.ylabel("normalized residual ")
            plt.xlabel("iterations")
            print("norm differences = {}".format(np.linalg.norm(u_sol - u_stat)))
            j= j + 1
    #plt.tight_layout()
    plt.show()
    

def sparse_sol():
    nu=0.01
    l=4
    m = 2**l -1
    A = fd_laplace(m,2)
    u_sol = np.random.rand(m**2)    
    A_2 = A.dot(A)
    S_jacobi = sparse.eye((m**2)) + nu * A_2            
    f =  S_jacobi.dot(u_sol)
    sparse.linalg.spsolve(S_jacobi, f)
    
def multi_sol():
    omega = 0.5
    nu1 = 3
    nu2 = 3
    nu=0.01
    level = 3
    l=4
    m = 2**l -1
    A = fd_laplace(m,2)    
    u_sol = np.random.rand(m**2)
    u_guess = np.zeros((m**2))    
    A_2 = A.dot(A)
    S_jacobi = sparse.eye((m**2)) + nu * A_2            
    f =  S_jacobi.dot(u_sol)
    multigrid_jacobi(nu, f, u_guess, m, omega, nu1, nu2, level)

def multigrid_vs_sparse():
    time=timeit.timeit("multi_sol()", setup="from multigrid_analysis import multi_sol")
    print(time)
#multigrid_vs_sparse()
#plot_mulitgrid_jacobi()
plot_mulitgrid_stat()


