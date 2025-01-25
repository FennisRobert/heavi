
import numpy as np
from enum import Enum
from ..component import BaseComponent
from ..rfcircuit import ComponentType, Network
from scipy.optimize import root_scalar

def coth(x):
    return -np.tan(x + np.pi/2)

def _microstrip_ereff(w: float, h: float, er: float, t: float):
    u = w/h
    T = t/h
    du1 = T/np.pi * np.log(1 + 4*np.exp(1)/(T*coth(6.517*u)**2))
    dur = 1/2 * (1 + 1/np.cosh(er -1 )) * du1
    u = u + dur
    a = 1 + (1/49) * np.log((u**4 + (u/52)**2)/(u**4 + 0.432)) + (1/18.7) * np.log(1 + (u/18.1)**3)
    b = 0.564 * ((er - 0.9) / (er + 3))**0.053
    return (er + 1)/2 + (er - 1)/2 * (1 + 10/u)**(-a*b)

def _microstrip_z0(w: float, h: float, er: float, t: float):
    # compute hammerstand and jensen
    u = w/h
    T = t/h
    du1 = T/np.pi * np.log(1 + 4*np.exp(1)/(T*coth(6.517*u)**2))
    dur = 1/2 * (1 + 1/np.cosh(er -1 )) * du1
    u = u + dur
    ereff = _microstrip_ereff(w, h, er, t)
    fu = 6 + (2 * np.pi - 6) * np.exp(-((30.666/u)**0.7528))
    Z0freespace = 376.73/np.sqrt(ereff)
    return Z0freespace / (2*np.pi) * np.log(fu/u + np.sqrt(1 + (2/u)**2))

def _stripline_ereff(w, h, er, t):
    return er

def _stripline_z0(w, h, er, t):
    Z0 = 60/np.sqrt(er) * np.log((4*(2*h + t))/(0.67*np.pi *(0.8*w + t)))
    return Z0

def _w_from_z0_microstrip(targetZ0: float, h: float, er: float, t: float):
    # Find the width of the microstrip line that gives the target impedance
    # Use _microstrip_z0 to find the impedance of the line by inverting the function using interpolation
    # Use the scipy.optimize.root_scalar function to find the root of the function
    # Return the width of the line

    def f(w):
        return _microstrip_z0(w, h, er, t) - targetZ0
    
    return root_scalar(f, bracket=[0.001, 10], xtol=0.006).root

def _w_from_z0_stripline(targetZ0: float, h: float, er: float, t: float):
    # Find the width of the stripline that gives the target impedance
    # Use _stripline_z0 to find the impedance of the line by inverting the function using interpolation
    # Use the scipy.optimize.root_scalar function to find the root of the function
    # Return the width of the line

    def f(w):
        return _stripline_z0(w, h, er, t) - targetZ0
    
    return root_scalar(f, bracket=[0.001, 10], xtol=0.006).root



class PCBStack:

    def __init__(self, network: Network, epsilon_r: float, tand: float, thickness: float, Nlayers: int):

        # Test that the layers are at least 2 and at most 10
        if Nlayers < 2 or Nlayers > 10:
            raise ValueError("The number of layers must be at least 2 and at most 10")
        
        # Test that the thickness is positive
        if thickness <= 0:
            raise ValueError("The thickness must be positive")
        
        # Test that the relative permittivity is positive
        if epsilon_r <= 0:
            raise ValueError("The relative permittivity must be positive")
        
        # Test that the loss tangent is between 0 and 1
        if tand < 0 or tand > 1:
            raise ValueError("The loss tangent must be between 0 and 1")
        
        self.network = network
        self.epsilon_r = epsilon_r
        self.tand = tand
        self.thickness = thickness
        self.Nlayers = Nlayers
