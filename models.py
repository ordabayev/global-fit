"""
Module containing fitting models.
"""
import numpy as np
from weeks import weeks
from wpar2 import wpar2
from laplace import Talbot

models = {}

def model(model_func):
    '''Add functions decorated with @model to the models list'''
    models[model_func.__name__] = model_func
    return model_func

@model
def nstep(t, n, k, A):
    """Sequential n-step unwinding
    A * k**n / (s * (k + s)**n)"""
    F = lambda s: A * k**n / (s * (k + s)**n)
    N = 64
    so, bo = wpar2(F,t[-1],N,0,30,30,50)
    y = weeks(F,t,N,so,bo)
    return y

@model
def nstep_kckend(t, A, n, kt, kd, kc, kend, r, C):
    """Sequential n-step translocation with two-step dissociation
    A/(1+n*r) * (1/(s+kc) * (1+kt*r/(s+kd)*(1-(kt/(s+kt+kd))**n))*(1+C*kc/(s+kend)))"""
    F = lambda s: A/(1+n*r) * (1/(s+kc) * (1+kt*r/(s+kd)*(1-(kt/(s+kt+kd))**n))*(1+C*kc/(s+kend))) 
    N = 64
    so, bo = wpar2(F,t[-1],N,0,30,30,50)
    y = weeks(F,t,N,so,bo)
    return y

@model
def nstep_kckend_talbot(t, A, n, kt, kd, kc, kend, r, C):
    """Sequential n-step translocation with two-step dissociation
    A/(1+n*r) * (1/(s+kc) * (1+kt*r/(s+kd)*(1-(kt/(s+kt+kd))**n))*(1+C*kc/(s+kend)))"""
    F = lambda s: A/(1+n*r) * (1/(s+kc) * (1+kt*r/(s+kd)*(1-(kt/(s+kt+kd))**n))*(1+C*kc/(s+kend))) 
    y = Talbot(F,t,N=50)
    return y

@model
def nstep_talbot(t, n, k, A, b):
    """Sequential n-step unwinding
    A * k**n / (s * (k + s)**n)"""
    F = lambda s: A * k**n / (s * (k + s)**n)
    y = Talbot(F,t,N=50) + b
    return y