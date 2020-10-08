#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:30:46 2020

@author: florianwolf
"""

import inspect
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt
from RMatrix import RMatrix
plt.close('all')

"""
TO DO:
    + implement fast poisson using the calculated eigenvalues 
    and eigenvectors and use it to compute its spectral radius
    (- implement poisson solver vectorized using only one loop)
    + plot behaviour of the methods for different m and nu
    - find nu for which Jacobi converges
    - compare both solvers to damped jacobi
    + calculate the conditional number based on nu
    
    - multigrid
"""


def fd_laplace(m, d):
    if d == 1:
        # build matrix blocks 
        four = -2*sparse.identity(m)
        onesUpper = sparse.eye(m, k=1)
        onesLower = sparse.eye(m, k=-1)
        T = four + onesUpper + onesLower

    elif d == 2:
        # use kronecker product to get 2D laplacian matrix
        four = -2*sparse.identity(m)
        onesUpper = sparse.eye(m, k=1)
        onesLower = sparse.eye(m, k=-1)
        T_bar = four + onesUpper + onesLower
        eye = sparse.eye(m)
        T = sparse.kron(eye, T_bar) + sparse.kron(T_bar, eye)
    return(T)


def fast_poisson(V, W, mu, lam, b):
    m = np.shape(V)[0]
    B = np.reshape(b, (m, m))
    B_tilde = ((W.T @((V.T @ B)).T)).T
    u_tilde = np.zeros((m, m))
    for i in np.arange(0, m):
        for j in np.arange(0,m):
            u_tilde[i,j] = (B_tilde[i,j])/(lam[i] + mu[j])
    U = (W@((V@u_tilde)).T).T
    u = np.reshape(U, m*m)
    return(u)

def get_system(m,nu):
    """
    Generated the left side system
    
    (nu*I + A^{-2})*u
    
    for a vector u. This is used to construct a linear operator
    to get a matrix free system
    """
    #A = fd_laplace(m,d=2)
    A_small = fd_laplace(m,d=1)
    eye_nu = nu*sparse.identity(m**2)
    lam, V = np.linalg.eigh(A_small.toarray())
    def left_side(v):
        sol = fast_poisson(V,V,lam,lam,v)
        sol2 = fast_poisson(V,V,lam,lam,sol)
        return(eye_nu.dot(v)+sol2)
    return(left_side)

def get_eigenvector(m,k):
    v = np.zeros(m)
    for i in range(0,m):
        v[i] = np.sin(((i+1) * np.pi * (k+1))/(m+1))
    return(v)

def laplace_small_decomposition(m):
    """
    Function to get the eigenvalues and eigenvectors of the
    small (1D) laplacian matrix using its toeplitz structure
    """
    lam = np.zeros(m)
    V = np.zeros((m,m))
    for k in range(0,m):
        lam[k] = (-2) + 2*np.cos((np.pi * (k+1))/(m+1))
        u = get_eigenvector(m,k)
        u = 1/np.linalg.norm(u)*u
        V[:,k] = u
        
    return(lam, V)

def laplace_eigs(m):
    lam_small, V_small = laplace_small_decomposition(m)

    lam = np.zeros(m**2)
    V = np.zeros((m**2, m**2))

    for i in range(0,m):
        for j in range(0,m):
            lam[j+i*m] = lam_small[i] + lam_small[j]
            V[:, j+i*m] = np.kron(V_small[:,i], V_small[:,j])
    return(lam, V)

def condition_number_normal(m, nu):
    """
    conditon number of nu*Id + A^{-2}
    """
    lam, V = laplace_eigs(m)
    lam = 1/(np.abs(lam)**2)
    sol = (nu + np.max(lam))/(nu + np.min(lam))
    return(sol)

def condition_number_factored(m, nu):
    """
    conditon number of nu*A^2 + 1
    """
    lam, V = laplace_eigs(m)
    lam = nu*(np.abs(lam)**2)
    sol = (1 + np.max(lam))/(1 + np.min(lam))
    return(sol)
    

def solver_stationary(u_guess,nu, y_d, f, m, maxIter=1000, tol=1e-6, disp=False):
    # get decomposition for fast poisson solver
    lam, V = laplace_small_decomposition(m)
    k = 1
    # get history of the norm of the residuals
    res_list = []
    
    # calculate the right hand side of the system
    right_side_f = fast_poisson(V, V, lam, lam, f) # =A^-1 *f
    right_side_f = fast_poisson(V, V, lam, lam, right_side_f) # =A^-2 *f
    right_side_y_d = fast_poisson(V, V, lam, lam, y_d) # = A^-1 * y_d
    right_side = right_side_y_d - right_side_f # = A^-1*(y_d- A^-1 *f)
    
    
    u_k = u_guess
    
    #construct linear operator of nu*Id + A^{-2}
    op = get_system(m,nu)
    operator = LinearOperator((m**2,m**2),op)
    
    res = np.linalg.norm(operator(u_k) - right_side)
    res_list.append(res)
    
    while res >= tol and k < maxIter:
        temp = fast_poisson(V, V, lam, lam, u_k)
        
        u_k = ((-1)*(1/nu) * fast_poisson(V, V, lam, lam, temp))+ 1/nu*right_side
        #print("u_k = {}".format(u_k))
        
        res = np.linalg.norm(operator(u_k) - right_side)
        res_list.append(res)
        
        k+=1
        
    return(u_k, k, res_list)

def solver_stationary_fixedRight(u_guess,nu, right_side, m, maxIter=500, tol=1e-6):
    # get decomposition for fast poisson solver
    lam, V = laplace_small_decomposition(m)
    k = 1
    # get history of the norm of the residuals
    res_list = []
    
    u_k = u_guess
    
    #construct linear operator of nu*Id + A^{-2}
    op = get_system(m,nu)
    operator = LinearOperator((m**2,m**2),op)
    
    res = np.linalg.norm(operator(u_k) - right_side)
    res_list.append(res)
    
    while res >= tol and k < maxIter:
        temp = fast_poisson(V, V, lam, lam, u_k)
        
        u_k = ((-1)*(1/nu) * fast_poisson(V, V, lam, lam, temp))+ 1/nu*right_side
        
        res = np.linalg.norm(operator(u_k) - right_side)
        res_list.append(res)
        
        k+=1
        
    return(u_k, k, res_list)

def solver_poisson_unfactored_cg(u_guess,nu, y_d, f, m, tol=1e-12, disp=False):
    """
    Solve the original system:
    
    (nu \cdot A^{-2})u = A^{-1} \cdot y_d - A^{-2}\cdot f
    
    This system is solved matrix free.
    """
    
    #initializing iteration count and residual history
    num_iters = 0
    res = []
    
    # create callback function to get residual history 
    # and the iteration counter
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
        frame = inspect.currentframe().f_back
        res.append(frame.f_locals['resid'])

    # create 1D and 2D laplacian matrices
    #A_small = fd_laplace(m, d=1)
    #A = fd_laplace(m, d=2)
    
    # get decomposition
    #lam_2, V_2 = np.linalg.eigh(A_small.toarray())
    lam, V = laplace_small_decomposition(m)
    
    # spectral_radius = get_spectralradius(lam, nu)
    # if disp:
    #     print("nu = {0}\nspectral radius = {1}\n".format(nu,spectral_radius))
    
    # calculate the right hand side of the system
    right_side_f = fast_poisson(V, V, lam, lam, f) # =A^-1 *f
    right_side_f = fast_poisson(V, V, lam, lam, right_side_f) # =A^-2 *f
    right_side_y_d = fast_poisson(V, V, lam, lam, y_d) # = A^-1 * y_d
    right_side = right_side_y_d - right_side_f # = A^-1*(y_d- A^-1 *f)
    
    #construct linear operator
    op = get_system(m,nu)
    operator = LinearOperator((m**2,m**2),op)
    
    # solve the system using the cg method
    u,info = cg(operator,right_side,x0=u_guess,tol=tol,callback=callback)
    return(u,info,num_iters,res)

        
def solver_poisson_factored_cg(u_guess, nu, y_d, f, m,tol=1e-12, disp=False):
    """
    Solve the by A^2 factorized system :
        
    (\nu * A^2 + \identity) \cdot u = A\cdot y_d - f 
    
    This system is solved matrix free.
    """
    #initializing iteration count and residual history
    num_iters = 0
    res = []
    
    # create callback function to get residual history 
    # and the iteration counter
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
        #if num_iters % 10 == 0:
        #    print("iter = {}".format(num_iters))
        frame = inspect.currentframe().f_back
        res.append(frame.f_locals['resid'])
    
    # init 1D and 2D laplacian matrices
    #A_small = fd_laplace(m, d=1)
    A = fd_laplace(m, d=2)
    lam, V = laplace_small_decomposition(m)


    # calculate the right hand side of problem
    right_side = A.dot(y_d)-f

    # define linear operator of the left side for the cg method
    def left_side(v):
        result = (nu*(A.dot(A))).dot(v)+(sparse.identity(m**2)).dot(v)
        return(result)
    left_side_op = LinearOperator((m**2, m**2), left_side)
    
    # spectral_radius = get_spectralradius_factored(lam, nu)
    # if disp:
    #     print("nu = {0}\nspectral radius = {1}\n".format(nu,spectral_radius))
    
    # solve system using cg method
    u,info = cg(left_side_op,right_side,x0=u_guess,tol=tol, callback=callback)
    
    return(u,info,num_iters,res)
    


def solver_damped_jacobi(A, u_guess, omega, f, maxIter,tol=1e-8):
    D_inverse = sparse.diags(1/(A.diagonal()),format="csr")
    m,m = D_inverse.shape
    k = 1
    res = []
    u_k = u_guess
    resi = (f - A.dot(u_k))
    resi_norm = np.linalg.norm(resi)
    res.append(resi_norm)
    while resi_norm >= tol and k < maxIter:
        u_k = u_k + omega* D_inverse.dot(resi)
        resi = f -  A.dot(u_k)
        resi_norm = np.linalg.norm(resi)
        res.append(resi_norm)
        k=k+1
    return(u_k, k, res)

def vcycle_jac(nu, nu1, nu2, m, u_guess, f, level, omega):
    A = fd_laplace(m, d=2)
    A_2 = np.dot(A,A)
    C = sparse.eye((m**2)) + nu * A_2
    omega_temp = omega
    if level == 1:
        u_sol = sparse.linalg.spsolve(C, f)
        res_norm = np.linalg.norm(f - C.dot(u_sol))
        return(u_sol, res_norm)
    else:
        ### PRE-SMOOTHING
        u_nu1,_ ,_ = solver_damped_jacobi(C, u_guess, omega_temp, f, nu1)
        ### RECURSION
        R = RMatrix(m)
        P = 2 * R.T
        
        res_temp = f - C.dot(u_nu1)
        
        f_new = R.dot(res_temp)
        
        m_new = int((m+1)/2 -1)
        ## !!!! init with zero vector !!!!
        u_init = np.zeros(m_new**2)
        
        u_temp, _ = vcycle_jac(nu, nu1, nu2, m_new, u_init, f_new, level-1, omega_temp)
        
        u_new = u_nu1 + P.dot(u_temp)
        ### POST-SMOOTHING        
        u_nu2,_ ,_ = solver_damped_jacobi(C, u_new, omega, f, nu2)
        res_norm = np.linalg.norm(f - C.dot(u_nu2))
        return(u_nu2, res_norm)
    
def multigrid_jacobi(nu, f, u_guess, m, omega, nu1, nu2, level):
    u_sol, res = vcycle_jac(nu, nu1, nu2, m, u_guess, f, level, omega)
    k = 1
    while res >= 1e-6 and k < 1000:
        u_sol, res = vcycle_jac(nu, nu1, nu2, m, u_sol, f, level, omega)
        k+=1
    return(u_sol, res, k)

def vcycle_stat(nu, nu1, nu2, m, u_guess, f, level):
    #construct linear operator
    op = get_system(m,nu)
    C = LinearOperator((m**2,m**2),op)
    if level == 1:
        system = np.zeros((m**2, m**2))
        for i in range(0, m**2):
            e = np.zeros(m**2)
            e[i] = 1
            system[i,:] = C.matvec(e)
        u_sol = np.linalg.solve(system, f) #######
        res_norm = np.linalg.norm(f - C(u_sol))
        return(u_sol, res_norm)
    else:
        ### PRE-SMOOTHING
        
        u_nu1,_ ,_ = solver_stationary_fixedRight(u_guess, nu, f, m, maxIter=nu1)
        ### RECURSION
        R = RMatrix(m)
        P = 2 * R.T
        
        res_temp = f - C(u_nu1)
        
        f_new = R.dot(res_temp)
        
        m_new = int((m+1)/2 -1)
        ## !!!! init with zero vector !!!!
        u_init = np.zeros(m_new**2)
        
        u_temp, _ = vcycle_stat(nu, nu1, nu2, m_new, u_init, f_new, level-1)
        
        u_new = u_nu1 + P.dot(u_temp)
        ### POST-SMOOTHING        
        u_nu2,_ ,_ = solver_stationary_fixedRight(u_new, nu, f, m, maxIter=nu2)
        res_norm = np.linalg.norm(f - C(u_nu2))
        return(u_nu2, res_norm)
    
def multigrid_stat(nu, f, u_guess, m, nu1, nu2, level):
    u_sol, res = vcycle_stat(nu, nu1, nu2, m, u_guess, f, level)
    k = 1
    while res >= 1e-6 and k < 1000:
        u_sol, res = vcycle_stat(nu, nu1, nu2, m, u_sol, f, level)
        k+=1
    return(u_sol, res, k)



