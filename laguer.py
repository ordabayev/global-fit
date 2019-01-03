# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 08:38:28 2018

@author: yerdos
"""

import numpy as np

#  The function p = laguer(a,x) evaluates a Laguerre polynomial
#  expansion with coefficients a at the values x, by using Clenshaw's
#  algorithm.   The vector x could be a scalar or vector, and the
#  coefficients in the vector a are ordered in increasing order of
#  the index.  

def laguer(a, x):
    N = len(a)-1
    
    unp1  =  np.zeros_like(x)
    un    =  a[N]*np.ones_like(x)
    
    for n in range(N,0,-1):
        unm1  =  (1/n)*(2*n-1-x)*un - n/(n+1)*unp1 + a[n-1]
        unp1  =  un
        un  =  unm1
    
    p = unm1
    return p




