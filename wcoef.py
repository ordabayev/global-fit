# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 08:15:19 2018

@author: yerdos
"""
import numpy as np

#function a = wcoef(F,N,sig,b)
#
#   The function wcoef.m computes the Weeks coefficients of
#   the function F(s) by using the midpoint version of the FFT.
#
def wcoef(F,N,sig,b):
    n = np.arange(-N,N,1)          # first compute the gridpoint for the FFT  ...       
    h = np.pi/N
    th = h * (n+1/2)
    y = b / np.tan(th/2)
    s = sig + y*np.array([1j]) 
    FF = F(s)             # ... then sample the function ... 
    FF = FF*(b+np.array([1j])*y)
    a = np.fft.fftshift(np.fft.fft(np.fft.fftshift(FF)))/(2*N)  # ... and apply the FFT.
    a = np.exp(np.array([-1j])*n*h/2)*a
    return a














