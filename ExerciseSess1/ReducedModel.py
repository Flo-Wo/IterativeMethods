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

def get_spectralradius(lam, nu):
    """
    Spectral radius for the system
    
    nu*I + A^{-2}
    
    Using the rules from
    https://de.wikipedia.org/wiki/Kronecker-Produkt#Rechenregeln
    to get that the eigenvalues of A (2D) are two times the eigenvalues
    of A (1D)
    """
    mu = (1/np.abs(2*lam))**2 #eigenvalues of A^{-2}
    return(nu + np.max(mu))

def get_spectralradius_factored(lam, nu):
    """
    Spectral radius for the system
    
    nu*A^2+ I
    
    Using the rules from
    https://de.wikipedia.org/wiki/Kronecker-Produkt#Rechenregeln
    to get that the eigenvalues of A (2D) are two times the eigenvalues
    of A (1D)
    """
    mu = nu * (np.abs(2*lam))**2
    return(1 + np.max(mu))


def get_conditional_number(lam, nu):
    mu = (1/np.abs(2*lam))**2 #eigenvalues of A^{-2}
    return((nu + np.max(mu))/(nu + np.min(mu)))

def get_conditional_number_factored(lam, nu):
    mu = nu * (np.abs(2*lam))**2
    return((1 + np.max(mu))/(1 + np.min(mu)))
    


def solver_poisson(u_guess,nu, y_d, f, m, tol=1e-12, disp=False):
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
    A_small = fd_laplace(m, d=1)
    #A = fd_laplace(m, d=2)
    
    # get decomposition
    #lam_2, V_2 = np.linalg.eigh(A_small.toarray())
    lam, V = laplace_small_decomposition(m)
    
    spectral_radius = get_spectralradius(lam, nu)
    if disp:
        print("nu = {0}\nspectral radius = {1}\n".format(nu,spectral_radius))
    
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

        
def solver_poisson_factored(u_guess, nu, y_d, f, m,tol=1e-12, disp=False):
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
    
    spectral_radius = get_spectralradius_factored(lam, nu)
    if disp:
        print("nu = {0}\nspectral radius = {1}\n".format(nu,spectral_radius))
    
    # solve system using cg method
    u,info = cg(left_side_op,right_side,x0=u_guess,tol=tol, callback=callback)
    
    return(u,info,num_iters,res)
    

def damped_jacobi():
    pass

