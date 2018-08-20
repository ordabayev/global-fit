# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 08:33:57 2018

@author: yerdos
"""

import numpy as np
from wcoef import wcoef
from laguer import laguer
from wpar2 import wpar2

#  http://appliedmaths.sun.ac.za/~weideman/research/weeks.html
#  The function f = weeks(F,t,N,sig,b) computes the inverse Laplace
#  transform f(t) of the transform F(s) by the method of Weeks.

#  Input:
#  F -- symbolic function of s.
#  t -- ordinate(s) where f(t) are to be computed (scalar or vector).
#  N -- number of terms in the Laguerre expansion.
#  sig, b -- free parameters in Weeks method.  b > 0, sig > sig_0.
#  Note: sig and b may be estimated by wpar1.m and wpar2.m.

#  Output:
#  f   -- function values at t.

#  Example of usage: >>F = '1./sqrt(s.^2+1)'; t = [0 1 2]; 
#                    >>f  = weeks(F, t, 16, 1, 1)

def weeks(F, t, N, sig, b):
    a   = np.real(wcoef(F,N,sig,b))        # Compute the coefficients.
    a   = a[N:2*N]                    # Extract a_n, n = 0,...,N-1.
    #t   = t[:]                          # Make sure t is a column vector.
    L   = laguer(a,2*b*t)               # Evaluate the Laguerre series. 
    f   = L*np.exp((sig-b)*t)             # Evaluate Weeks expansion.
    return f

#def F(s):
#   return 1/np.sqrt(s**2+1)
#    return np.exp(-4*np.sqrt(s))

#N = 32; t = 1#np.array([0.1, 1, 10])
#sig = 0.7; b = 1.75
#f = weeks(Ftest,t,N,sig,b)
#so, bo = wpar2(F,t,N,0,30,30,50)
#print(so, bo)
#f = weeks(F,t,N,so,bo)

#t = np.array([0.1, 1, 10])
#f  = weeks(Ftest, t, 16, 1, 1)
#print(f)