# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 16:55:36 2018

@author: yerdos
"""

from scipy.optimize import fminbound
from werr2e import werr2e
from werr2t import werr2t

#  The function [so,bo] = wpar2(F,t,N,sig0,sigmax,bmax,Ntol) estimates 
#  the optimal parameters s (= sigma) and b for the Weeks method.  
#  The function requires no information on the singularities of the transform.
#
#  Input:
#  F      -- Transform; symbolic function of s.
#  t      -- Ordinate where f(t) are to be computed (scalar).
#  N      -- Number of terms in the Weeks expansion.
#  sig0   -- Laplace convergence asbscissa.
#  sigmax -- Maximum possible value of sigma.
#  bmax   -- Maximum possible value of b.
#  Ntol   -- Determines the tolerance in the optimization routine fmin.
#            Large Ntol => small tolerance.  Recommended: Ntol = 20 to 50.
#  Output:
#  (so, bo) -- Estimated parameters to be used in weeks.m or weekse.m.

def wpar2(F, t, N, sig0, sigmax, bmax, Ntol):
    tols = (sigmax-sig0)/Ntol 
    tolb = bmax/Ntol
    
    so = fminbound(werr2e,sig0,sigmax,args=(F,t,N,sig0,sigmax,bmax,tolb),xtol=tols)
    bo = fminbound(werr2t,0,bmax,args=(F,N,so),xtol=tolb) 
    return so, bo

