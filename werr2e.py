# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 17:08:47 2018

@author: yerdos
"""

import numpy as np
from scipy.optimize import fminbound
from werr2t import werr2t
from wcoef import wcoef

#   The function E = errorb(sig,F,t,N,sig0,sigmax,cc,tolb) computes
#   the error bound E = truncation plus conditioning error
#   on the optimal curve b = b(sig).

def werr2e(sig,F,t,N,sig0,sigmax,bmax,tolb):
    b  = fminbound(werr2t,0,bmax,args=(F,N,sig),xtol=tolb)  # Estimate optimal b=bopt(sig)
    
    M  = 2*N
    a  = wcoef(F,M,sig,b)
    a1 = a[2*N:3*N]
    sa1 = np.sum(np.abs(a1))
    a2 = a[3*N:4*N]
    sa2 = np.sum(np.abs(a2))
    E  = np.exp(sig*t) * (sa2+np.finfo(float).eps*sa1)
    E  = np.log(E)
    return E









