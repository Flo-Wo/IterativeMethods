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

def fd_laplace(m, d):
    if d == 1:
        # build matrix blocks 
        four = -2*sparse.identity(m)
        onesUpper = sparse.eye(m, k=1)
        onesLower = sparse.eye(m, k=-1)
        T = four + onesUpper + onesLower

    elif d == 2:
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

def fun1(m,nu):
    A = fd_laplace(m,d=2)
    A_small = fd_laplace(m,d=1)
    eye_nu = nu*sparse.identity(m**2)
    lam, V = np.linalg.eigh(A_small.toarray())
    def fun2(v):
        sol = fast_poisson(V,V,lam,lam,v)
        sol2 = fast_poisson(V,V,lam,lam,sol)
        return(eye_nu.dot(v)+sol2)
    return(fun2)
        
def solver_poisson_factored(u_guess, nu, y_d, f, m,tol=1e-12):
    """
    Solve by A^2 factorized system :
        
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
        frame = inspect.currentframe().f_back
        res.append(frame.f_locals['resid'])
    
    # init 1D and 2D laplacian matrices
    #A_small = fd_laplace(m, d=1)
    A = fd_laplace(m, d=2)

    # calculate the right hand side of problem
    right_side = A.dot(y_d)-f

    # define linear operator of the left side for the cg method
    def left(v):
        result = (nu*(A.dot(A))).dot(v)+(sparse.identity(m**2)).dot(v)
        return(result)
    left_operator = LinearOperator((m**2, m**2), left)
    
    # solve system using cg method
    u,info = cg(left_operator,right_side,x0=u_guess,tol=tol, callback=callback)
    
    return(u,info,num_iters,res)
    
def solver_poisson(u_guess,nu, y_d, f,m, tol=1e-12):
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
    lam, V = np.linalg.eigh(A_small.toarray())
    # calculate the right hand side of the system
    right_side_f = fast_poisson(V, V, lam, lam, f) # =A^-1 *f
    right_side_f = fast_poisson(V, V, lam, lam, right_side_f) # =A^-2 *f
    right_side_y_d = fast_poisson(V, V, lam, lam, y_d) # = A^-1 * y_d
    right_side = right_side_y_d - right_side_f # = A^-1*(y_d- A^-1 *f)
    
    #construct linear operator
    op = fun1(m,nu)
    operator = LinearOperator((m**2,m**2),op)
    
    # solve the system using the cg method
    u,info = cg(operator,right_side,x0=u_guess,tol=tol,callback=callback)
    return(u,info,num_iters,res)

m = 10
nu = 10000


y_d = 10*np.ones(m**2)

u_sol = np.ones(m**2)


A = fd_laplace(m,d=2)
f  = (-1)*(nu* A.dot(A.dot(u_sol)) + u_sol - A.dot(y_d))
#print("f = {}".format(f))

u_guess = np.random.rand(m**2)

#u_guess[0] = 1

u_test,info,num_iters,res = solver_poisson_factored(u_guess, nu, y_d, f,m)

x = np.arange(0,num_iters)
plt.semilogy(x,res)
plt.title('Konvergenzverhalten')

print("Number of iterations = {}\n".format(num_iters))


print("||u_sol - u_test|| = {}\n".format(np.linalg.norm(u_test-u_sol)))
