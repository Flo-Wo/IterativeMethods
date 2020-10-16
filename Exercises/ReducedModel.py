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
Authors: Florian Wolf, Michael Thiele, Thanh-Van Huynh
Date: 09.10.2020

This file contains multiple solvers for an optimal control problem including
    - fast poisson solver framework to fastly calculate the inverse of our
    matrix system
    - stationary solver
    - damped jacobi solver
    - matrix free krylov method using the conjugated gradient method
    - multigrid with the stationary method and damped jacobi as smoothers

We wrote this file as a project of the lecture "iterative methods and 
preconditioning" from Jun.-Prof. Gabriele Ciaramella, held in the WS2020/2021
at the University of Konstanz.

The other according files contain multiple tests and functions to plot
the behaviour of our solvers for different parameters. We used them
to analyse their convergence behaviour. 

"""

def create_norm(m):
    """
    Function to generate our discretized norm based on the parameter m

    Parameters
    ----------
    m : integer
        number of grid points used

    Returns
    -------
    norm : callable
        norm(x) = h**2 * ||x||_2

    """
    h = 1/(m+1)
    def norm(v):
        sol = h * np.linalg.norm(v)
        return(sol)
    return(norm)

def fd_laplace(m, d):
    """
    Fuction to generate the laplace matrix for dimension one and two

    Parameters
    ----------
    m : positive int
        size the matrix should have (for d=2 the matrix size is m**2 x m**2), 
        regarding your desired discretization
    d : integer with values one or two
        dimension parameter

    Returns
    -------
    Laplace matrix for your desired dimension and size

    """
    h = 1/(m+1)
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
        # T = kron(T_1, I) + kron(I, T_1)
        T = sparse.kron(eye, T_bar) + sparse.kron(T_bar, eye)
    return(1/(h**2) * T)


def fast_poisson(V, W, mu, lam, b):
    """
    Fast poisson solver framework to solve equations of the type
    Au = b where the matrix A is of the form
    
        A = kron(I, A_1) + kron(A_2, I)
    
    where kron represents the kronecker product. You need to give the 
    eigenvector and eigenvalue decomposition of A_1 and A_2

    Parameters
    ----------
    V : ndarray
        matrix containing normalized eigenvectors of A_1
        to the corresponding eingevalues lam
    W : ndarray
        matrix containing normalized eigenvectors of A_2
        to the corresponding eingevalues lam
    mu : ndarray
        eigenvalues of A_1 to the corresponding eigenvectors V
    lam : ndarray
        eigenvalues of A_1 to the corresponding eigenvectors W
    b : ndarray
        vector representing the right-hand side of the system
        Au=b

    Returns
    -------
    u : ndarray
        solution vector of the system

    """
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

    Parameters
    ----------
    m : positive integer
        size of the matrix system
    nu : positive real number
        regularization parameter of the optimal control problem

    Returns
    -------
    left_side : callable
        function representing the left-hand side of our equation
    """
    A = fd_laplace(m,d=2)
    # A_small = fd_laplace(m,d=1)
    # lam, V = np.linalg.eigh(A_small.toarray())
    eye_nu = nu*sparse.identity(m**2)
    lam, V = laplace_small_decomposition(m)
    def left_side(v):
        sol = sparse.linalg.spsolve(A, v)
        sol2 = sparse.linalg.spsolve(A, sol)
        #sol = fast_poisson(V,V,lam,lam,v)
        #sol2 = fast_poisson(V,V,lam,lam,sol)
        return(eye_nu.dot(v)+sol2)
    return(left_side)



def get_eigenvector(m,k):
    """
    Function to return the k-th eigenvector of the
    one-dimensional laplace matrix

    Parameters
    ----------
    m : positive integer
        size of the matrix and therefore size of the vector
    k : positive integer
        index k of the k-th eigenvector

    Returns
    -------
    v : ndarray
        k-th eigenvector of the 1D laplace matrix

    """
    v = np.zeros(m)
    for i in range(0,m):
        v[i] = np.sin(((i+1) * np.pi * (k+1))/(m+1))
    return(v)

def laplace_small_decomposition(m):
    """
    Function to get the eigenvalues and eigenvectors of the
    small (1D) laplacian matrix using its toeplitz structure

    Parameters
    ----------
    m : positive integer
        size of the matrix

    Returns
    -------
    lam : ndarray
        vector including all eigenvalues of the matrix
    V : ndarray
        matrix including all normalized eigenvectors of the matrix
        with respect to their corresponding eigenvalue in the vector lam

    """
    h = 1/(m+1)
    lam = np.zeros(m)
    V = np.zeros((m,m))
    for k in range(0,m):
        lam[k] = 1/(h**2) * ((-2) + 2*np.cos((np.pi * (k+1))/(m+1)))
        u = get_eigenvector(m,k)
        u = 1/np.linalg.norm(u)*u
        V[:,k] = u
        
    return(lam, V)

def laplace_eigs(m):
    """
    Function to get the eigenvectors and eigenvalues of the 2D-laplace matrix

    Parameters
    ----------
    m : positive integer
        size of the matrix

    Returns
    -------
    lam : ndarray
        vector containing all eigenvalues
    V : ndarray
        matrix including normalized eigenvectors

    """
    lam_small, V_small = laplace_small_decomposition(m)

    lam = np.zeros(m**2)
    V = np.zeros((m**2, m**2))

    for i in range(0,m):
        for j in range(0,m):
            lam[j+i*m] = lam_small[i] + lam_small[j]
            V[:, j+i*m] = np.kron(V_small[:,i], V_small[:,j])
            V[:, j+i*m] = 1/np.linalg.norm(V[:, j+i*m]) * V[:, j+i*m]
    return(lam, V)

def condition_number_normal(m, nu):
    """
    Function to calculate the conditon number of nu*Id + A^{-2} using 
    the calculated eigenvalues

    Parameters
    ----------
    m : positive integer
        size of the matrix
    nu : positive real number
        regularization parameter of the optimal control problem

    Returns
    -------
    sol : positive real number
        conditon number of the system

    """
    lam, V = laplace_eigs(m)
    lam = 1/(np.abs(lam)**2)
    sol = (nu + np.max(lam))/(nu + np.min(lam))
    return(sol)

def condition_number_factored(m, nu):
    """
    Function to calculate the conditon number of the factored system nu*A^2 + 1}
    using the calculated eigenvalues

    Parameters
    ----------
    m : positive integer
        size of the matrix
    nu : positive real number
        regularization parameter of the optimal control problem

    Returns
    -------
    sol : positive real number
        conditon number of the system

    """
    lam, V = laplace_eigs(m)
    lam = nu*(np.abs(lam)**2)
    sol = (1 + np.max(lam))/(1 + np.min(lam))
    return(sol)
    

def solver_stationary(u_guess,nu, y_d, f, m, maxIter=1000, tol=1e-8):
    """
    Function to use a stationary solver for the initial optimality system
    using the iteration process
    
        u_{n+1} = -1/nu A^{-2} u_n + 1/nu (A^-1*(y_d- A^-1 *f))
    

    Parameters
    ----------
    u_guess : ndarray
        inital guess the solver should use as a starting point/first iteration
    nu : positive real number
        regularization paramter
    y_d : ndarray
        vector representing the vector y_d of the optimal control problem
    f : ndarray
        vector representing the vector y_d of the optimal control problem
    m : positive integer
        size of the matrix control how fine the discretization should be
    maxIter : positive integer, optional
        maximum number of iterations. The default is 1000.
    tol : positive real number, optional
        tolerance used to compare the residuals to. The default is 1e-8.

    Returns
    -------
    u : ndarray
        solution of the system
    k : integer
        number of iterations needed
    res_list : list
        list with the history of the norm of the residuals

    """
    norm = create_norm(m)
    
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
    
    # append norm of the first residual
    res = norm(operator(u_k) - right_side)
    res_list.append(res)
    
    while res >= tol and k < maxIter:
        # solving -1/nu A^{-2} u_n + 1/nu right_side
        temp = fast_poisson(V, V, lam, lam, u_k)
        
        u_k = ((-1)*(1/nu) * fast_poisson(V, V, lam, lam, temp))+ 1/nu*right_side
        # append norm of the residual
        res = norm(operator(u_k) - right_side)
        res_list.append(res)
        
        k+=1
        
    return(u_k, k, res_list)

def solver_stationary_fixedRight(u_guess,nu, right_side, m, maxIter=500, tol=1e-8):
    """
    This is a method to use the iteration of the stationary solver for an
    arbtrary right-hand side of the equation. It is used to solve systems of
    the form
    
        (nu*Id + A^{-2})*u = right_side
    
    and used as a smoother for multigrid method (here the right-hand side
    differs from cycle to cycle).

    Parameters
    ----------
    u_guess : ndarray
        initial guess the algorithm should use to start its iterations
    nu : positive real number
        regularization parameter
    right_side : ndarray
        right-hand side of the desired equation
    m : positive integer
        size of the matrix
    maxIter : positive integer, optional
        maximum number of iterations. The default is 500.
    tol : positive real number, optional
        tolerance for the algorithm, to compare the residuals to.
        The default is 1e-8.

    Returns
    -------
    u : ndarray
        solution vector
    k : integer
        number of iterations needed
    res_list : list
        list with the history of the norm of the residuals

    """
    norm = create_norm(m)
    # get decomposition for fast poisson solver
    lam, V = laplace_small_decomposition(m)
    k = 1
    # get history of the norm of the residuals
    res_list = []
    
    u_k = u_guess
    
    #construct linear operator of nu*Id + A^{-2}
    op = get_system(m,nu)
    operator = LinearOperator((m**2,m**2),op)
    # caclculate the first norm of the residual
    res = norm(operator(u_k) - right_side)
    res_list.append(res)
    
    A = fd_laplace(m, d=2)
    
    while res >= tol and k < maxIter:
        temp = sparse.linalg.spsolve(A, u_k)
        
        #temp = fast_poisson(V, V, lam, lam, u_k)
        
        # solve current iteration
        #temp2 = fast_poisson(V, V, lam, lam, temp)
        temp2 = sparse.linalg.spsolve(A, temp)
        u_k = ((-1)*(1/nu) * temp2)+ 1/nu*right_side
        # update history of the residuals
        res = norm(operator(u_k) - right_side)
        res_list.append(res)
        
        k+=1
        
    return(u_k, k, res_list)

def solver_poisson_unfactored_cg(u_guess, nu, y_d, f, m, tol=1e-8):
    """
    Method to solve the initial optimality system 
    
        (nu \cdot A^{-2})u = A^{-1} \cdot y_d - A^{-2}\cdot f
    
    using a matrix free operator and the cg method.

    Parameters
    ----------
    u_guess : ndarray
        inital guess used by cg
    nu : positive real number
        regularization parameter
    y_d : ndarray
        desired state/function
    f : ndarray
        vector/function of the condition in the optimality system
    m : positive integer
        size of the matrix/system used
    tol : positive real number, optional
        tolerance for the cg algoritm. The default is 1e-8.

    Returns
    -------
    u : ndarray
        solution vector
    info : integer
        info whether the algorithm worked properly returned by cg
        0  : successful exit 
        >0 : convergence to tolerance not achieved, number of iterations 
        <0 : illegal input or breakdown
    num_iters : integer
        number of iterations needed from cg
    res : list
        history with the norm of the residuals

    """    
    #initializing iteration count and residual history
    num_iters = 0
    res = []
    h = 1/(m+1)
    
    # create callback function to get residual history 
    # and the iteration counter from the build in cg-method
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
        frame = inspect.currentframe().f_back
        res.append(h**2 * frame.f_locals['resid'])
    
    # get decomposition
    lam, V = laplace_small_decomposition(m)
    
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
    return(u, info, num_iters, res)

        
def solver_poisson_factored_cg(u_guess, nu, y_d, f, m,tol=1e-8):
    """
    Method to solve the factored optimality system 
    
        (\nu * A^2 + \identity) \cdot u = A\cdot y_d - f 
    
    using a matrix free operator and the cg method.

    Parameters
    ----------
    u_guess : ndarray
        inital guess used by cg
    nu : positive real number
        regularization parameter
    y_d : ndarray
        desired state/function
    f : ndarray
        vector/function of the condition in the optimality system
    m : positive integer
        size of the matrix/system used
    tol : positive real number, optional
        tolerance for the cg algoritm. The default is 1e-8.

    Returns
    -------
    u : ndarray
        solution vector
    info : integer
        info whether the algorithm worked properly returned by cg
        0  : successful exit 
        >0 : convergence to tolerance not achieved, number of iterations 
        <0 : illegal input or breakdown
    num_iters : integer
        number of iterations needed from cg
    res : list
        history with the norm of the residuals

    """ 
    #initializing iteration count and residual history
    num_iters = 0
    res = []
    
    h = 1/(m+1)
    
    # create callback function to get residual history 
    # and the iteration counter
    def callback(xk):
        nonlocal num_iters
        num_iters += 1
        frame = inspect.currentframe().f_back
        res.append(h**2 * frame.f_locals['resid'])
    
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
    
    
    # solve system using cg method
    u,info = cg(left_side_op,right_side,x0=u_guess,tol=tol, callback=callback)
    
    return(u, info, num_iters, res)
    


def solver_damped_jacobi(A, u_guess, omega, f, maxIter,tol=1e-8, norm=()):
    """
    Function to solve a system 
    
        Au = f
    
    using the damped jacobi method

    Parameters
    ----------
    A : ndarray/sparse matrix
        matrix of the system
    u_guess : ndarray
        inital guess the algorithm should use to start with
    omega : real number in (0,1)
        damping parameter for the algorithm
    f : ndarray
        right-hand side of the system
    maxIter : int
        maximum number of iterations the algorithm should do
    tol : positive real number, optional
        tolerance the algorithm should use for the residuals.
        The default is 1e-8.

    Returns
    -------
    u : ndarray
        solution for the system
    k : integer
        number of iterations needed
    res : list
        list with the history of the norm of the residuels

    """
    
    if norm == ():
        def norm(v):
            return(np.linalg.norm(v))
    
    # get diagonal matrix in csr form, to manipulate its entries
    D_inverse = sparse.diags(1/(A.diagonal()),format="csr")
    # get size of the matrix
    m,m = D_inverse.shape
    k = 1
    # init empty history of the norm of the residuals
    res = []
    u_k = u_guess
    resi = (f - A.dot(u_k))
    resi_norm = norm(resi)
    # get norm of the first residual
    res.append(resi_norm)
    while resi_norm >= tol and k < maxIter:
        # one step of the damped jacobi 
        u_k = u_k + omega* D_inverse.dot(resi)
        # calculate the norm of the current residual
        resi = f -  A.dot(u_k)
        resi_norm = norm(resi)
        res.append(resi_norm)
        k=k+1
    return(u_k, k, res)

def vcycle_jac(nu, nu1, nu2, m, u_guess, f, level, omega):
    """
    Function to run one vcycle of the multigrid method using jacobi as a
    smoother. In this case we try to solve the initial system.

    Parameters
    ----------
    nu : positive real number
        regularization parameter
    nu1 : integer
        number of pre-smoothing steps used
    nu2 : integer
        number of post-smoothing steps used
    m : integer
        size of the current matrix
    u_guess : ndarray
        initial guess of the current run
    f : ndarray
        function of the current right-hand side of the system
    level : int
        maximum number of levels the algorithm should do in the recursion
    omega : real number in (0,1)
        damping parameter for the smoother

    Returns
    -------
    u : ndarray
        solution after the last post-smoothing step
    res_norm : list
        list with the norm of the residuals

    """
    
    norm = create_norm(m)
    h = 1/(m+1)
    # construct matrix to get the right-hand side of the system
    A = fd_laplace(m, d=2)
    A_2 = np.dot(A,A)
    C = sparse.eye((m**2)) + nu * A_2
    # save omega for the recursion
    omega_temp = omega
    if level == 1:
        # on the last level the system gets solved with the build-in solver
        u_sol = sparse.linalg.spsolve(C, f)
        res_norm = norm(f - C.dot(u_sol))
        return(u_sol, res_norm)
    else:
        ### PRE-SMOOTHING
        u_nu1,_ ,_ = solver_damped_jacobi(C, u_guess, omega_temp, f, nu1, norm=norm)
        ### RECURSION
        # restriction matrix (full weighted restriction matrix) to get
        # the smaller system
        R = RMatrix(m)
        # projection matrix, to get back to the full system
        P = 4 * R.T
        
        res_temp = f - C.dot(u_nu1)
        
        f_new = R.dot(res_temp)
        # new size of the smaller system
        m_new = int((m+1)/2 -1)
        ## init with zero vector
        u_init = np.zeros(m_new**2)
        
        u_temp, _ = vcycle_jac(nu, nu1, nu2, m_new, u_init, f_new, level-1, omega_temp)
        # project the current solution into the higher dimensional space
        u_new = u_nu1 + P.dot(u_temp)
        ### POST-SMOOTHING        
        u_nu2,_ ,_ = solver_damped_jacobi(C, u_new, omega, f, nu2)
        res_norm = norm(f - C.dot(u_nu2))
        return(u_nu2, res_norm)


def multigrid_jacobi(nu, f, u_guess, m, omega, nu1, nu2, level,maxIter=1000, tol=1e-6):
    """
    Function to run use multigrid as a solver with damped jacobi as a 
    smoother. In this case we try to solve the initial system.

    Parameters
    ----------
    nu : positive real number
        regularization parameter
    nu1 : integer
        number of pre-smoothing steps used
    nu2 : integer
        number of post-smoothing steps used
    m : integer
        size of the current matrix
    u_guess : ndarray
        initial guess of the current run
    f : ndarray
        function of the current right-hand side of the system
    level : int
        maximum number of levels the algorithm should do in the recursion
    omega : real number in (0,1)
        damping parameter for the smoother

    Returns
    -------
    u : ndarray
        solution after the last post-smoothing step
    res_norm : list
        list with the norm of the residuals
    k : int
        number of iterations neede

    """
    norm = create_norm(m)
    u_sol = u_guess
    k = 1
    h = 1/(m+1)
    A = fd_laplace(m, d=2)
    A_2 = np.dot(A,A)
    C = sparse.eye((m**2)) + nu * A_2
    u_sol=u_guess
    f_new = f
    res_his = []
    res = norm( f - C.dot(u_sol)) 
    res_his.append(res)
    while res >= tol and k < maxIter  and res <= 10e10:
        u_temp, res = vcycle_jac(nu, nu1, nu2, m, u_sol, f_new, level, omega)
        u_sol = u_sol + u_temp
        f_new = f - C.dot(u_sol)
        res_his.append(res)
        k+=1
    return(u_sol, res_his, k)

def vcycle_stat(nu, nu1, nu2, m, u_guess, f, level):
    """
    Function to run one vcycle of the multigrid method using the stationary 
    method as a smoother. In this case we try to solve the initial system.

    Parameters
    ----------
    nu : positive real number
        regularization parameter
    nu1 : integer
        number of pre-smoothing steps used
    nu2 : integer
        number of post-smoothing steps used
    m : integer
        size of the current matrix
    u_guess : ndarray
        initial guess of the current run
    f : ndarray
        function of the current right-hand side of the system
    level : int
        maximum number of levels the algorithm should do in the recursion

    Returns
    -------
    u : ndarray
        solution after the last post-smoothing step
    res_norm : list
        list with the norm of the residuals

    """
    norm = create_norm(m)
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
        res_norm = norm(f - C(u_sol))
        return(u_sol, res_norm)
    else:
        ### PRE-SMOOTHING
        
        u_nu1,_ ,_ = solver_stationary_fixedRight(u_guess, nu, f, m, maxIter=nu1)
        ### RECURSION
        # restriction matrix (full weighted restriction matrix) to get
        # the smaller system
        R = RMatrix(m)
        # projection matrix, to get back to the full system
        P = 4 * R.T
        
        res_temp = f - C(u_nu1)
        
        f_new = R.dot(res_temp)
        # get new matrix size
        m_new = int((m+1)/2 -1)
        ## init with zero vector
        u_init = np.zeros(m_new**2)
        
        u_temp, _ = vcycle_stat(nu, nu1, nu2, m_new, u_init, f_new, level-1)
        # project the current solution into the higher dimensional space
        u_new = u_nu1 + P.dot(u_temp)
        ### POST-SMOOTHING        
        u_nu2,_ ,_ = solver_stationary_fixedRight(u_new, nu, f, m, maxIter=nu2)
        res_norm = norm(f - C(u_nu2))

        return(u_nu2, res_norm)


def multigrid_stat(nu, f, u_guess, m, nu1, nu2, level, maxIter=50,tol=1e-12):
    """
    Function to run use multigrid as a solver with the stationary method as a 
    smoother. In this case we try to solve the initial system.

    Parameters
    ----------
    nu : positive real number
        regularization parameter
    nu1 : integer
        number of pre-smoothing steps used
    nu2 : integer
        number of post-smoothing steps used
    m : integer
        size of the current matrix
    u_guess : ndarray
        initial guess of the current run
    f : ndarray
        function of the current right-hand side of the system
    level : int
        maximum number of levels the algorithm should do in the recursion

    Returns
    -------
    u : ndarray
        solution after the last post-smoothing step
    res_norm : list
        list with the norm of the residuals
    k : int
        number of iterations neede

    """
    norm = create_norm(m)
    u_sol, res = vcycle_stat(nu, nu1, nu2, m, u_guess, f, level)
    u_sol=u_guess
    op = get_system(m,nu)
    C = LinearOperator((m**2,m**2),op)
    res  = norm(f - C(u_sol))
    res0 = res
    k = 1
    res_his = []
    res_his.append(res/res0)
    while res/res0 >= tol and k < maxIter and res <= 10e10:
        u_sol, res = vcycle_stat(nu, nu1, nu2, m, u_sol, f, level)
        k+=1
        res_his.append(res/res0)
    return(u_sol, res_his, k)
