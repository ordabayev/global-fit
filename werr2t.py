# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 17:11:59 2018

@author: yerdos
"""

import numpy as np
from wcoef import wcoef

#  The function T = werr2t(b,F,N,sig) computes the 
#  estimate for the truncation error T in the Weeks method.

def werr2t(b, F, N, sig):
    M   = 2*N
    a   = wcoef(F,M,sig,b)
    sa2 = np.sum(np.abs(a[3*N:4*N]))
    T   = np.log(sa2)
    return T

