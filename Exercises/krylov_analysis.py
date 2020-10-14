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

# generate tikz picture from the plot
import tikzplotlib

def plot_convergenceRate():
    plt.figure(figsize = (30, 20))
    #plt.suptitle(r"Convergence analysis of cg-method using different values of $m$ and $\nu$")
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
            plt.ylabel("norm of residual")
            plt.xlabel("number of iterations")
            plt.title(r"$m =$ {0}, $\nu =$ {1}".format(m, nu))
            j+=1
    plt.tight_layout()
    plt.show()

def plot_conditionNumber():
    plt.figure(figsize = (15, 25))
    j = 1
    for l in range(4,7):
        cond = []
        cond_factored = []
        nu_values = []
        m = 2**l-1
        for k in range(4):
            nu = 0.01 * 10**k
            nu_values.append(nu)
            cond.append(condition_number_normal(m, nu))
            cond_factored.append(condition_number_factored(m, nu))
        plt.subplot(2,4,j)
        plt.semilogx(nu_values, cond,"r-", label="initial system")
        plt.semilogx(nu_values, cond_factored,"b-", label="factored system")
        plt.xticks(nu_values)
        plt.yscale("log")
        plt.legend(loc="upper right")
        plt.title(r"$m =$ {}".format(m))
        plt.xlabel(r"values of $\nu$")
        plt.ylabel("condition number of the system")
        j+=1
    plt.tight_layout()
    plt.show()
 
plot_convergenceRate()

