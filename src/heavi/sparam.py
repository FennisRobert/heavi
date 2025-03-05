import numpy as np
from typing import Callable
import re
from scipy.interpolate import interpn, RegularGridInterpolator
from loguru import logger

def frange(fmin: float, fmax: float, n: int) -> np.ndarray:
    """Generate n frequencies from fmin to fmax"""
    return np.linspace(fmin, fmax, n)

class Sparameters:


    def __init__(self, S: np.ndarray, f: np.ndarray):
        """ S-parameters object
        
        Parameters
        ----------
        S : np.ndarray
            Scattering matrix of shape (nports, nports, nfreqs) 
        s"""
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

    def __init__(self, interp: RegularGridInterpolator):
        self._interp: RegularGridInterpolator = interp

    def __call__(self, *args):
        return np.squeeze(self._interp(np.meshgrid(*args, indexing='ij')))

class NDSparameters(Sparameters):

    def __init__(self, mdim_data: np.ndarray, axes: list[np.ndarray]):
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
        if any([ax.shape[0] > 1 for ax in self._outer_axes]):
            logger.warning('Parallel axis found in outer dimensions. Defaulting to first detected axis')
        print([ax[0,:].shape for ax in self._outer_axes], self._axes[-1].shape)
        axes = [np.squeeze(ax[0,:]) for ax in self._outer_axes] + [self._axes[-1],]
        return Interpolator(RegularGridInterpolator(axes, self.S(p1, p2), bounds_error=False, fill_value=None))
    