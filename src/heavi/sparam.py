########################################################################################
##
##    An S-parameter data container class
##    This module contains classes for handling S-parameter data
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################

#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------


import numpy as np
from typing import Callable
import re
from scipy.interpolate import interpn, RegularGridInterpolator
from loguru import logger


#  ___            __  ___    __        __  
# |__  |  | |\ | /  `  |  | /  \ |\ | /__` 
# |    \__/ | \| \__,  |  | \__/ | \| .__/ 
# -------------------------------------------


def frange(fmin: float, fmax: float, n: int) -> np.ndarray:
    """Generates a linearly spaced frequency range

    Args:
        fmin (float): Lower frequency bound
        fmax (float): Upper frequency bound
        n (int): Number of points

    Returns:
        np.ndarray: Linearly spaced frequency range
    """    
    return np.linspace(fmin, fmax, n)


#  __             __   __   ___  __  
# /  ` |     /\  /__` /__` |__  /__` 
# \__, |___ /~~\ .__/ .__/ |___ .__/ 
# -------------------------------------------

class Sparameters:


    def __init__(self, S: np.ndarray, f: np.ndarray):
        """ S-parameters object
        
        Parameters
        ----------
        S : np.ndarray
            Scattering matrix of shape (nports, nports, nfreqs) 
        """
        self._S: np.ndarray = S
        self.f: float = f
        self.nports: int = S.shape[0]
        self.nfreqs: int = S.shape[2]
    
    @property
    def shape(self) -> tuple:
        return self._S.shape
    
    def S(self, p1: int, p2: int):
        """Get S-parameter S[p1, p2] counting from 1, S(1,1) is S11"""
        
        # check if p1 and p2 are valid ports
        if p1 < 1 or p1 > self.nports:
            raise ValueError(f"Port {p1} out of range")
        if p2 < 1 or p2 > self.nports:
            raise ValueError(f"Port {p2} out of range")
        return self._S[p1-1, p2-1, :]
    
    def __call__(self, p1: int, p2: int):
        return self.S(p1, p2)
    
    def __getattr__(self, name):
        match = re.match(r'^S(\d+)(\d+)$', name)
        if match:
            p1, p2 = int(match.group(1)), int(match.group(2))
            return self.S(p1, p2)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __dir__(self):
        # Return dynamically generated attributes for autocompletion
        valid_s_parameters = [f"S{m}{n}" for m in range(1, 6) for n in range(1, 5)]
        return valid_s_parameters + super().__dir__()
    

    ### All S-parameters of a 5 port network for autocompletion purposes.
    @property
    def S11(self):
        return self.S(1, 1)

    @property
    def S12(self):
        return self.S(1, 2)

    @property
    def S13(self):
        return self.S(1, 3)

    @property
    def S14(self):
        return self.S(1, 4)

    @property
    def S15(self):
        return self.S(1, 5)

    @property
    def S21(self):
        return self.S(2, 1)

    @property
    def S22(self):
        return self.S(2, 2)

    @property
    def S23(self):
        return self.S(2, 3)

    @property
    def S24(self):
        return self.S(2, 4)

    @property
    def S25(self):
        return self.S(2, 5)

    @property
    def S31(self):
        return self.S(3, 1)

    @property
    def S32(self):
        return self.S(3, 2)

    @property
    def S33(self):
        return self.S(3, 3)

    @property
    def S34(self):
        return self.S(3, 4)

    @property
    def S35(self):
        return self.S(3, 5)

    @property
    def S41(self):
        return self.S(4, 1)

    @property
    def S42(self):
        return self.S(4, 2)

    @property
    def S43(self):
        return self.S(4, 3)

    @property
    def S44(self):
        return self.S(4, 4)

    @property
    def S45(self):
        return self.S(4, 5)

    @property
    def S51(self):
        return self.S(5, 1)

    @property
    def S52(self):
        return self.S(5, 2)

    @property
    def S53(self):
        return self.S(5, 3)

    @property
    def S54(self):
        return self.S(5, 4)

    @property
    def S55(self):
        return self.S(5, 5)


class Interpolator:
    """Wrapper class for scipy RegularGridInterpolator
    """    
    def __init__(self, interp: RegularGridInterpolator):
        self._interp: RegularGridInterpolator = interp

    def __call__(self, *args) -> np.ndarray:
        """Interpolates the data at the given points

        Args:
            *args: N-dimensional points to interpolate at
        
        Returns:
            np.ndarray: Interpolated data
        """        
        return np.squeeze(self._interp(np.meshgrid(*args, indexing='ij')))

class StatSparam:

    def __init__(self, S: np.ndarray):
        self._S: np.ndarray = S
        self.ntrials: int = S.shape[0]

    @property
    def amplitude_mean(self) -> np.ndarray:
        return np.abs(self._S).mean(axis=0)
    
    @property
    def amplitude_std(self) -> np.ndarray:
        return np.abs(self._S).std(axis=0)
    
    @property
    def power_mean(self) -> np.ndarray:
        return (np.abs(self._S)**2.0).mean(axis=0)
    
    @property
    def power_std(self) -> np.ndarray:
        return (np.abs(self._S)**2.0).std(axis=0)
    
    @property
    def rms_mean(self) -> np.ndarray:
        return np.sqrt(np.abs(self._S)**2.0).mean(axis=0) 
    
    @property
    def rms_std(self) -> np.ndarray:
        return np.sqrt(np.abs(self._S)**2.0).std(axis=0)


class StatSparameters(Sparameters):
    """Series of S-parameters over multiple Monte Carlo runs
    """

    def __init__(self, Ntrials: int):
        self.Ntrials: int = Ntrials
        self._Sdata: list[np.ndarray] = []

    @property
    def nports(self) -> int:
        if len(self._Sdata) == 0:
            return 0
        return self._Sdata[0].shape[0]
    
    @property
    def nfreqs(self) -> int:
        if len(self._Sdata) == 0:
            return 0
        return self._Sdata[0].shape[2]
    
    def add(self, S: np.ndarray):
        """Add S-parameter data to the container

        Args:
            S (np.ndarray): S-parameter data of shape (nports, nports, nfreqs)
        """        
        if S.shape != (self.nports, self.nports, self.nfreqs):
            raise ValueError(f"Invalid S-parameter shape. Expected {(self.nports, self.nports, self.nfreqs)}, got {S.shape}")
        self._Sdata.append(S)
    
    def S(self, p1: int, p2: int) -> StatSparam:
        """Get S-parameter S[p1, p2] counting from 1, S(1,1) is S11

        Args:
            p1 (int): Port number
            p2 (int): Port number
        
        Returns:
            np.ndarray: S-parameter data
        """        
        return StatSparam(np.stack([S[p1-1, p2-1, :] for S in self._Sdata], axis=1))
    
    def __call__(self, p1: int, p2: int) -> StatSparam:
        return self.S(p1, p2)
    
    

class NDSparameters(Sparameters):
    """N-dimensional S-parameter data container class
    """

    def __init__(self, mdim_data: np.ndarray, axes: list[np.ndarray]):
        """Initializes the N-dimensional S-parameter container

        S-parameter data is expected to be of shape (..., nports, nports, nfreqs)


        Args:
            mdim_data (np.ndarray): _description_
            axes (list[np.ndarray]): _description_
        """        
        self._sdata: np.ndarray = mdim_data
        self._axes: list[np.ndarray] = axes
        self._shape: tuple[int] = self._sdata.shape
        self._extra_dims: int = len(self._shape) - 3
        self.nports: int = self._shape[-2]
        self.nfreqs: int = self._shape[-1]

    @property
    def _outer_axes(self) -> tuple:
        return tuple([ax for ax in self._axes[:self._extra_dims]])
    
    @property
    def _outer_slice(self) -> tuple:
        return tuple([slice(None) for _ in range(self._extra_dims)])
    
    def S(self, p1: int, p2: int) -> np.ndarray:
        """Get S-parameter S[p1, p2] counting from 1, S(1,1) is S11"""
        # check if p1 and p2 are valid ports
        if p1 < 1 or p1 > self.nports:
            raise ValueError(f"Port {p1} out of range")
        if p2 < 1 or p2 > self.nports:
            raise ValueError(f"Port {p2} out of range")
        return np.squeeze(self._sdata[self._outer_slice + (p1-1, p2-1, None)])
    
    def __call__(self, i1: int, i2: int) -> np.ndarray:
        return self.S(i1, i2)
    
    def intS(self, p1: int, p2: int) -> Interpolator:
        """Interpolates the S-parameter data for the given port combinations"

        Args:
            p1 (int): Port number
            p2 (int): Port number
        
        Returns:
            Interpolator: Interpolator object.

        Example:
        --------
        >>> S = NDSparameters(data, axes)
        >>> S.intS(1, 2)(f1, f2)
        """
        if any([ax.shape[0] > 1 for ax in self._outer_axes]):
            logger.warning('Parallel axis found in outer dimensions. Defaulting to first detected axis')
        axes = [np.squeeze(ax[0,:]) for ax in self._outer_axes] + [self._axes[-1],]
        return Interpolator(RegularGridInterpolator(axes, self.S(p1, p2), bounds_error=False, fill_value=None))
    