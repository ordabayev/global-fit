# laplace.py with numpy

# Talbot suggested that the Bromwich line be deformed into a contour that begins
# and ends in the left half plane, i.e., z \to \infty at both ends.
# Due to the exponential factor the integrand decays rapidly
# on such a contour. In such situations the trapezoidal rule converge
# extraordinarily rapidly.
# For example here we compute the inverse transform of F(s) = 1/(s+1) at t = 1
#
# >>> error = Talbot(1,24)-exp(-1)
# >>> error
#   (3.3306690738754696e-015+0j)
#
# Talbot method is very powerful here we see an error of 3.3e-015
# with only 24 function evaluations
#
# Created by Fernando Damian Nieuwveldt      
# email:fdnieuwveldt@gmail.com
# Date : 25 October 2009
#
# Adapted to numpy arrays by Yerdos Ordabayev
# email: yerdos2030@gmail.com
# Date : 10 February 2018
#
# Reference
# L.N.Trefethen, J.A.C.Weideman, and T.Schmelzer. Talbot quadratures
# and rational approximations. BIT. Numerical Mathematics,
# 46(3):653 670, 2006.

import numpy as np

def Talbot(F,t,N=24):
    
#   Initiate the stepsize
    h = 2*np.pi/N;
  
#   Shift contour to the right in case there is a pole on the positive real axis : Note the contour will
#   not be optimal since it was originally devoloped for function with
#   singularities on the negative real axis
#   For example take F(s) = 1/(s-1), it has a pole at s = 1, the contour needs to be shifted with one
#   unit, i.e shift  = 1. But in the test example no shifting is necessary

    shift = 0.0;
    ans =   0.0;
    
    if (t == 0).any():
        print("ERROR:   Inverse transform can not be calculated for t=0")
        return ("Error");
        
#   The for loop is evaluating the Laplace inversion at each point theta which is based on the trapezoidal   rule
    for k in range(0,N):
        theta = -np.pi + (k+1./2)*h;
        z = shift + N/t*(0.5017*theta/np.tan(0.6407*theta) - 0.6122 + np.array([0.2645j])*theta); 
        dz = N/t*(-0.5017*0.6407*theta/np.sin(0.6407*theta)**2+0.5017/np.tan(0.6407*theta)+np.array([0.2645j]));
        ans = ans + np.exp(z*t)*F(z)*dz;
        
    return ((h/(np.array([2j])*np.pi))*ans).real
      

# testfunction: Laplace-transform of exp(-t)  
def F(s):
    return 1.0/(s+1.0)