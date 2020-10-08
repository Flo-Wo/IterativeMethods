#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 11:30:00 2020

@author: jaekle
"""
import numpy as np
from scipy import sparse

def RMatrix(m):
    """
    RMATRIX construct restriction matrix of size M^2 x m^2
      R=RMatrix(m) constructs a restriction matrix of size M^2 x m^2
      to be used in a multigrid code.

    Parameters
    ----------
    m : int
        dimension.
        
    Returns
    -------
    R : sparse
        restriction matrx for multigrid code.

    """    
    
    e = np.ones(m**2)                      # diagonal entries
    e1 = np.tile(np.hstack((np.ones(m-1),0)),m)
    e2 = np.ones(m**2)
    B = np.zeros((m**2,3**2))
    B[0:len(e1),0] = e1/16
    B[0:len(e2),1] = e2/8        
    B[0:len(e1),2] = np.flipud(e1/16)  
    B[0:len(e1),3] = e1/8
    B[0:len(e),4] = e/4
    B[0:len(e1),5] = np.flipud(e1/8)     # flipud for spdiags convention 
    B[0:len(e1),6] = e1/16
    B[0:len(e2),7] = np.flipud(e2/8)
    B[0:len(e1),8] = np.flipud(e1/16)
    
    diags = np.array([-m-1, -m, -m+1, -1, 0, 1, m-1, m, m+1])
    R = sparse.spdiags(B.T,diags,m**2,m**2,'lil') 
    inner = np.arange(1,m**2+1).reshape((m,m)).T
    G = np.zeros((m+2,m+2))                # extract even gridpoints
    G[1:-1,1:-1] = inner
    id_ = G[2:m:2,2:m:2]
    R = R[id_.T.flatten()-1,:]
    
    return R
