from ..numeric import SimParam
from ..component import BaseComponent
from ..rfcircuit import Network, Node
from ..sparam import Sparameters
from typing import Callable
from scipy.optimize import minimize
import numpy as np
from enum import Enum

class OptimVar(SimParam):

    def __init__(self, initial_value: float, bounds: tuple[float, float] = None, mapping: Callable = None):
        self._value = initial_value
        self.bounds = bounds
        self.mapping: Callable = mapping

    def set(self, value: float):
        self._value = value

    def __call__(self, f: np.ndarray) -> np.ndarray:
        if self.mapping is None:
            return self._value*np.ones_like(f)
        else:
            return self.mapping(self._value)*np.ones_like(f)

class LpNorm(Enum):
    MIN = 0
    MAX = 1
    LP1 = 2
    LP2 =3

    def get_metric(self):
        if self is self.MAX:
            return np.max
        if self is self.MIN:
            return np.min
        if self is self.LP1:
            return np.mean
        if self is self.LP2:
            return lambda x: np.sqrt(np.mean(x**2))

def pnorm(norm: float) -> Callable:
    def func(x):
        return np.mean(x**norm)**(1/norm)
    return func

def dBbelow(dB_level: float,
            norm: float | LpNorm = LpNorm.MAX):
    if isinstance(norm, LpNorm):
        meval = norm.get_metric()
    else:
        meval = pnorm(norm)
    
    def metric(S: np.ndarray) -> float:
        return meval(np.clip(20*np.log10(np.abs(S))-dB_level, a_min=0, a_max=None))/5

    return metric

class FreqRequirement:

    def __init__(self, 
                 fmin: float,
                 fmax: float,
                 nF: int,
                 parameter: tuple[int, int],
                 metric: Callable,
                 weight: float = 1):
        self.fs = np.linspace(fmin,fmax,nF)
        self.metric: Callable = metric
        self.param: tuple[int, int] = parameter
        self.slc: slice = None
        self.weight: float = weight

    def eval(self, S: Sparameters) -> float:
        value = self.metric(S.S(self.param[0],self.param[1])[self.slc])*self.weight
        return value

def gen_fill_area(fmin, fmax, below: float = None, above: float = None):
    if below is not None:
        return (fmin, fmax, below,0)
    if above is not None:
        return (fmin, fmax, -100, above)
        

class Optimizer:

    def __init__(self, network: Network):
        self.network = network
        self.parameters: list[OptimVar] = []
        self.requirements: list[FreqRequirement] = []

    @property
    def bounds(self) -> list[tuple[float, float]]:
        return [p.bounds for p in self.parameters]
    
    @property
    def x0(self) -> np.ndarray:
        return np.array([p.value for p in self. parameters])

    def add_param(self, initial, bounds: tuple[float, float] = None, mapping: Callable = None) -> OptimVar:
        param = OptimVar(initial, bounds=bounds, mapping=mapping)
        self.parameters.append(param)
        return param
    
    def cap(self, logscale: bool = True) -> OptimVar:
        if not logscale:
            return self.add_param(1e-12, (1e-13,1e-6))
        else:
            return self.add_param(-12, (-13,-6), mapping= lambda x: 10**(x))
    
    def ind(self, logscale: bool = True) -> OptimVar:
        if not logscale:
            return self.add_param(1e-9, (1e-12,1e-5))
        else:
            return self.add_param(-9, (-12,-5), mapping= lambda x: 10**(x))
    
    def add_goal(self, 
                 fmin: float,
                 fmax: float,
                 n_frequencies: int,
                 indices: tuple[int,int],
                 metric: Callable,
                 weight: float = 1):
        self.requirements.append(FreqRequirement(fmin, fmax, n_frequencies, indices, metric, weight=weight))

    def generate_objective(self, 
                           pnorm=2, 
                           initial: np.ndarray = None, 
                           differential_weighting: float = 0,
                           differential_weighting_exponent: float = 5) -> Callable:

        if initial is not None:
            for p, v in zip(self.parameters, initial):
                p.value = v
        n = 0
        fs = []
        for req in self.requirements:
            fs = fs + list(req.fs)
            req.slc = slice(n,len(fs))
            n = len(fs)
        fs = np.array(fs)
        NR = len(self.requirements)
        if differential_weighting > 0:
            def objective(coeffs):
                for p,c in zip(self.parameters, coeffs):
                    p.set(c)
                S = self.network.run(fs)
                Ms = np.array([abs(req.eval(S)) for req in self.requirements])
                #print(Ms, max(Ms)-min(Ms))
                return (np.mean(Ms**pnorm))**(1/pnorm) + differential_weighting*(max(Ms)-min(Ms))**differential_weighting_exponent
        else:
            def objective(coeffs):
                for p,c in zip(self.parameters, coeffs):
                    p.set(c)
                S = self.network.run(fs)
                subobj = [abs(req.eval(S))**pnorm for req in self.requirements]
                return (sum(subobj)/NR)**(1/pnorm)
               
        return objective
    



