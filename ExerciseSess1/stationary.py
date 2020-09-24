#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 14:30:46 2020

@author: florianwolf
"""

import numpy as np
from scipy import sparse


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

m = 10

A_1 = fd_laplace(m, d=1)
A_2 = fd_laplace(m, d=2)
lam, V = np.linalg.eigh(A_1.toarray())
x = np.random.rand(m**2)
b = A_2.dot(x)


def fast_poisson(V, W, mu, lam, b):
    m = np.shape(V)[0]
    B = np.reshape(b, (m, m))
    #print("B = {}\n".format(B))
    B_tilde = ((W.T @((V.T @ B)).T)).T
    #B_tilde_2 = ((W.T).dot(((V.T).dot(B)).T)).T
    #B_tilde_2 = (np.linalg.inv(V).dot((np.linalg.inv(V).dot(B)).T)).T
    #print("B_tilde = {}\n".format(B_tilde))
    #print("Norm B_tilde - B = {}\n".format(np.linalg.norm(B_tilde - B_tilde_2)))
    u_tilde = np.zeros((m, m))
    for i in np.arange(0, m):
        for j in np.arange(0,m):
            u_tilde[i,j] = (B_tilde[i,j])/(lam[i] + mu[j])
    #print("u_tilde = {}\n".format(u_tilde))
    U = (W@((V@u_tilde)).T).T
    #U_2 = (W.dot((V.dot(u_tilde)).T)).T
    #print("Norm U - U_2= {}\n".format(np.linalg.norm(U - U_2)))
    u = np.reshape(U, m*m)
    #print("u = {}\n".format(u))
    return(u)

print("#--------------- Solver section ---------------#")
x_sol = fast_poisson(V, V, lam, lam, b)
print("Norm Difference fast poisson: \n{}\n".format(np.linalg.norm(x-x_sol)))

def solver_poisson(u_guess, nu, y_d, f, m, test, tol=10^-6, maxIter = 5):
    A_small = fd_laplace(m, d=1)
    A = fd_laplace(m, d=2)
    
    # get decomposition
    lam, V = np.linalg.eigh(A_small.toarray())
    # solve right hand side using poisson solver
    right_side_f = fast_poisson(V, V, lam, lam, f) # =A^-1 *f
    right_side_f = fast_poisson(V, V, lam, lam, right_side_f) # =A^-2 *f
    right_side_y_d = fast_poisson(V, V, lam, lam, y_d) # = A^-1 * y_d
    right_side = right_side_y_d - right_side_f # = A^-1*(y_d- A^-1 *f)
    print("\nNorm difference right hand sides \n= {}\n".format(np.linalg.norm(right_side - test)))
    #right_side = test
    
    #print("right side = {}".format(right_side))
    
    # solve system iteratively using the splitting M = A^-2 and N = (-1)*nu*I
    # and the residual/correction form
    
    u = u_guess
    #res = right_side - ((nu*I - A^-2)u_0)
    
    eye_nu = nu*sparse.identity(m**2) #=nu*I
    
    # update residual
    u_res = fast_poisson(V, V, lam, lam, u)
    u_res_2 = fast_poisson(V, V, lam, lam, u_res) # = A^-2
    
    res = right_side - (eye_nu.dot(u) + u_res_2)
    k = 0
    print("k = {0}, \nu = {1}\n".format(k,u))
    while np.linalg.norm(res) >= tol and k < maxIter:
        # update u with formula u_k+1 = u_k + M^-1 r_k with M^-1 = A^2
        u = u + A.dot(A.dot(res))
        # update residual
        u_res = fast_poisson(V, V, lam, lam, u) #=A^-1*u
        u_res_2 = fast_poisson(V, V, lam, lam, u_res) # = A^-2*u
        res = right_side - (eye_nu.dot(u) + u_res_2) # residual_k = f - (nu*Id + A^-2)u_k
        
        # update counter
        k += 1
        print("k = {0}, \n u = {1}\n".format(k,u))
    return(u, k)


m = 3
nu = 0.5


y_d = 10*np.ones(m**2)

u_sol = np.ones(m**2)

######## Testing stuff with fixed right hand side ############
A_small = fd_laplace(m, d=1)
A = fd_laplace(m, d=2)

lam, V = np.linalg.eigh(A_small.toarray())

u_A_first = fast_poisson(V, V, lam, lam, u_sol)

u_A_second = fast_poisson(V, V, lam, lam, u_A_first)

eye_nu = nu*sparse.identity(m**2) #=nu*I

right_hand_side = eye_nu.dot(u_sol) + u_A_second

######## END ############

f  = (-1)*(nu* A.dot(A.dot(u_sol)) + u_sol - A.dot(y_d))
#print("f = {}".format(f))

u_guess = np.zeros(m**2)
#u_guess[0] = 1

print("#--------------- Algorithm section ---------------#")

u_test, it = solver_poisson(u_guess, nu, y_d, f, m, test=right_hand_side)

print("#--------------- Result section ---------------#")

print("\nu_sol = {}\n".format(u_sol))

print("u_test = {}\n".format(u_test))

print("Norm of difference = {}\n".format(np.linalg.norm(u_test-u_sol)))


