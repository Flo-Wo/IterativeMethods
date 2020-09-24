#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np

from scipy.sparse import issparse

#==============================================================================
#                                 Conjugate Gradient
#==============================================================================

def cg(A, f, u_0=(), eps=10e-10, maxiter = 10**5):
    """
    Solve the linear system Au=f with the Conjugate Gradient method.
    Input matrix A can be a 2-D-array or an sparse matrice

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (n,n)
        The square matrix A for solving the linear system Ax=f.
    f : ndarray, shape (n,)
        The vector representing the right hand side of the equation. 
    u_0 : ndarray, shape (n,), optional
        Initial start. If not choosen the algorithm start with zeros. 
        The default is ().
    eps : float, optional
        Stopping criteria. The algorithm stops if norm(e_k+1) < eps, where
        
            e_k+1 = inv(M)*N*e_k. 
            
        The default is 1e-10.
    maxiter : int, optional
        Stopping criteria. If Maxiter is reached, the algorithm stops even if 
        the stopping criteria for the tolerance is not reached. 
        The default is 10**5.

    Returns
    -------
    u_k : ndarray, shape (n,)
        The solution vector.
    count : int
        Number of iterations.
        
    Examples
    --------
    Solving a linear system where A is given as the discrete Laplacian 
    operator computed with finite differences. For tests, this is implemented
    in oppy.itMet.FDLaplacian
    
    >>> from oppy.itMet import FDLaplacian
    
    The minimizing method was chosen as the conjugate gradient method
    
    >>> from oppy.itMet import cg
    
    To construct a test we use numpy arrays
    
    >>> import numpy as np
    
    Set up test case
    
    The following computes the 2 dimensional discrete Laplacian
    
    >>> m = 100
    >>> d = 2
    >>> A = FDLaplacian(m,d)
    
    Notice, in that case, that A is a scipy.sparse matrix.
    Choosing a random solution and computing the right hand side
    
    >>> x = np.random.rand(m**d)
    >>> b = A.dot(x)
    
    Now, we can solve our system
    
    >>> x_sol, count = cg(A,b)
    
    and compare the solution to the exact one
    
    >>> print(np.linalg.norm(x-x_sol))
    >>> 4.98418709648239e-07
    
    Notice, depending on your system, this number could be a bit different but 
    the error should be something arround 1e-7.
    """
    
    #check if the input A is sparse/dense
    if issparse(A) == True:
        def action(x):
            return A.dot(x)
    elif type(A) == np.ndarray:
        def action(x):
            return A.dot(x)
    else:
        #test if the input is a function which gives the action of Ax
        test = A(f)
        if len(test) == len(f):
            def action(x):
                return A(x)
        else:
            print('wrong input for A')
            return
        
    # define u_0 if necessary
    if u_0 == ():
        u_0 = np.zeros(len(f))
        
    #define r_0
    r_k = f-action(u_0)
    #defin p_0
    p_k = r_k
    count = 0
    #define u_0
    u_k = u_0
    #cause we need the norm more often than one time, we save it
    tmp_r_k_norm = np.linalg.norm(r_k)
    while count < maxiter and tmp_r_k_norm > eps:
        #save the matrix vector product
        tmp = action(p_k)
        #calculate alpha_k
        alpha_k = (tmp_r_k_norm)**2/(np.dot(p_k,tmp))
        #calculate u_k+1
        u_k = u_k + alpha_k*p_k
        #calculate r_k+1
        r_k = r_k - alpha_k*tmp
        #save the new norm of r_k+1
        tmp_r_k1 = np.linalg.norm(r_k)
        #calculate beta_k
        beta_k = (tmp_r_k1)**2/(tmp_r_k_norm)**2
        tmp_r_k_norm = tmp_r_k1
        #calculate p_k+1
        p_k = r_k + beta_k*p_k
        count += 1
    return u_k, count