from ..numeric import SimParam
from ..rfcircuit import Network
from ..sparam import Sparameters
from typing import Callable
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod

class OptimVar(SimParam):
    """ Class for defining an optimisation variable. 
    
    This class is used to define a variable that can be optimised. It is a subclass of SimParam, and can be used in the same way.
    
    Parameters
    ----------
    initial_value : float
        The initial value of the variable.
    bounds : tuple[float, float], optional
        The bounds of the variable. If not specified, the variable is unbounded.
    mapping : Callable, optional
        A function that maps the variable to a different space. If not specified, the variable is used as
        is. The function should take a float and return a float.
    """
    def __init__(self, initial_value: float, 
                 bounds: tuple[float, float] = None, 
                 mapping: Callable = None,
                 unit: str = ''):
        self._value = initial_value
        self.bounds = bounds
        self.mapping: Callable = mapping
        self.unit: str = unit

    def set(self, value: float):
        self._value = value

    def __call__(self, f: np.ndarray) -> np.ndarray:
        if self.mapping is None:
            return self._value*np.ones_like(f)
        else:
            return self.mapping(self._value)*np.ones_like(f)

class PNorm(Enum):
    """ Enum for defining the type of norm to use in the optimisation.
    
    This enum is used to define the type of norm to use in the optimisation. It can be used to specify the type of norm to use in the optimisation.
    
    Attributes
    ----------
    MIN : int
        The minimum norm.
    MAX : int
        The maximum norm.
    P1 : int
        The P1 norm (average).
    P2 : int
        The P2 norm (root mean square).
    """
    MIN = 0
    MAX = 1
    P1 = 2
    P2 =3

    def get_metric(self) -> Callable:
        """ Get the metric function for the norm."""
        if self is self.MAX:
            return np.max
        if self is self.MIN:
            return np.min
        if self is self.P1:
            return np.mean
        if self is self.P2:
            return lambda x: np.sqrt(np.mean(x**2))

def generalized_mean(norm: float) -> Callable:
    """ Generate a generalized mean function.
    
    This function generates a generalized mean function with the specified norm.
    
    Parameters
    ----------
    norm : float
        The norm of the mean.
    
    Returns
    -------
    Callable
        The generalized mean function.
        """
    def func(x):
        return np.mean(x**norm)**(1/norm)
    return func


def dBbelow(dB_level: float,
            norm: float | PNorm = PNorm.P2) -> Callable:
    """ Generate a function that calculates the dB level below a specified level.
    
    This function generates a function that calculates the dB level below a specified level.
    
    Parameters
    ----------
    dB_level : float
        The dB level to calculate the level below.
    norm : float | PNorm, optional
        The norm to use in the calculation. If not specified, the P2 norm is used.
    
    Returns
    -------
    Callable
        The function that calculates the dB level below the specified level.
    """

    if isinstance(norm, PNorm):
        meval = norm.get_metric()
    else:
        meval = generalized_mean(norm)
    
    def metric(S: np.ndarray) -> float:
        return meval(np.clip(20*np.log10(np.abs(S))-dB_level, a_min=0, a_max=None))/10
    return metric

def dBabove(dB_level: float,
            norm: float | PNorm = PNorm.P2) -> Callable:
    """ Generate a function that calculates the dB level above a specified level.
    
    This function generates a function that calculates the dB level above a specified level.
    
    Parameters
    ----------
    dB_level : float
        The dB level to calculate the level above.
    norm : float | PNorm, optional
        The norm to use in the calculation. If not specified, the P2 norm is used.
    
    Returns
    -------
    Callable
        The function that calculates the dB level above the specified level.
    """
    
    if isinstance(norm, PNorm):
        meval = norm.get_metric()
    else:
        meval = generalized_mean(norm)
    
    def metric(S: np.ndarray) -> float:
        return meval(np.clip(dB_level - 20*np.log10(np.abs(S)), a_min=0, a_max=None))/10
    
    return metric


class Requirement(ABC):
    """ Abstract class for defining a requirement.
    
    This class is used to define a requirement for an optimiser. It is an abstract class, and should be subclassed to define a specific requirement.
    
    """
    @abstractmethod
    def eval(self, S: Sparameters) -> float:
        pass


class FrequencyLevel(Requirement):

    def __init__(self, 
                 fmin: float,
                 fmax: float,
                 nF: int,
                 sparam: tuple[int, int],
                 upper_limit: float = None,
                 lower_limit: float = None,
                 weight: float = 1,
                 f_norm: float | PNorm = 4):
        self.fs = np.linspace(fmin,fmax,nF)
        self.param: tuple[int, int] = sparam
        self.slc: slice = None
        self.weight: float = weight

        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

        if upper_limit is not None and lower_limit is not None:
            self.metric = dBbelow(upper_limit, norm=f_norm) + dBabove(lower_limit, norm=f_norm)
        elif upper_limit is not None:
            self.metric = dBbelow(upper_limit, norm=f_norm)
        elif lower_limit is not None:
            self.metric = dBabove(lower_limit, norm=f_norm)
        else:
            raise ValueError("At least one of upper_limit or lower_limit must be specified")
        

    def eval(self, S: Sparameters) -> float:
        value = self.metric(S.S(self.param[0],self.param[1])[self.slc])*self.weight
        return value
    
    def generate_fill_area(self) -> tuple[float, float, float, float]:
        lower = self.lower_limit
        upper = self.upper_limit
        if lower is None:
            lower = 0
        if upper is None:
            upper = -100
        return (self.fs[0], self.fs[-1], upper, lower)


class Optimiser:
    """ Class for defining an optimiser.
    
    This class is used to define an optimiser for a network. It is used to define the parameters to optimise, the requirements to meet, and the objective function to optimise.
    
    Parameters
    ----------
    network : Network
        The network to optimise.
    
    """
    def __init__(self, network: Network):
        self.network = network
        self.parameters: list[OptimVar] = []
        self.requirements: list[FrequencyLevel] = []

    @property
    def spec_area(self) -> list[tuple[float, float, float, float]]:
        """ Get the fill areas for the requirements."""
        return [req.generate_fill_area() for req in self.requirements]
    
    @property
    def bounds(self) -> list[tuple[float, float]]:
        """ Get the bounds of the optimisation variables."""
        return [p.bounds for p in self.parameters]
    
    @property
    def x0(self) -> np.ndarray:
        """ Get the initial values of the optimisation variables."""
        return np.array([p.value for p in self. parameters])

    def add_param(self, initial, 
                  bounds: tuple[float, float] = None, 
                  mapping: Callable = None,
                  unit: str = '') -> OptimVar:
        """ Add an optimisation variable.
        
        This method adds an optimisation variable to the optimiser.
        
        Parameters
        ----------
        initial : float
            The initial value of the variable.
        bounds : tuple[float, float], optional
            The bounds of the variable. If not specified, the variable is unbounded.
        mapping : Callable, optional
            A function that maps the variable to a different space. If not specified, the variable is used as
            is. The function should take a float and return a float.
        
        Returns
        -------
        OptimVar
            The optimisation variable.
        """
        param = OptimVar(initial, bounds=bounds, mapping=mapping, unit=unit)
        self.parameters.append(param)
        return param
    
    def cap(self, 
            logscale: bool = True, 
            initial: float = -12,
            limits: tuple[float, float] = (-13, -6)) -> OptimVar:
        """ Add a capacitor parameter.
        
        This method adds a capacitor parameter to the optimiser.
        The default capacitor range is from 0.1 pF to 1 uF.
        
        Parameters
        ----------
        logscale : bool, optional
            Whether to use a log scale for the parameter. If not specified, a log scale is used.
        initial : float, optional
            The initial value of the parameter. If not specified, the default value is used (1 pF).
        limits : tuple[float, float], optional
            The limits of the parameter. If not specified, the default limits are used (0.1 pF to 1 uF).
        
        Returns
        -------
        OptimVar
            The capacitor parameter.
        """
        if not logscale:
            return self.add_param(10**initial, (1*10**limits[0], 1*10**limits[1]), unit='f')
        else:
            return self.add_param(initial, limits, mapping= lambda x: 10**(x), unit='f')
    
    def ind(self, 
            logscale: bool = True,
            initial: float = -9,
            limits: tuple[float, float] = (-12,-5)) -> OptimVar:
        """ Add an inductor parameter.
        
        This method adds an inductor parameter to the optimiser.
        
        Parameters
        ----------
        logscale : bool, optional
            Whether to use a log scale for the parameter. If not specified, a log scale is used.
        
        Returns
        -------
        OptimVar
            The inductor parameter.
        """
        if not logscale:
            return self.add_param(10**initial, (10**(limits[0]),10**(limits[1])), unit='H')
        else:
            return self.add_param(initial, limits, mapping= lambda x: 10**(x), unit='H')
    
    def add_requirement(self, req: Requirement):
        """ Add a requirement to the optimiser.
        
        This method adds a requirement to the optimiser.
        
        Parameters
        ----------
        req : Requirement
            The requirement to add.
        """
        self.requirements.append(req)

    def add_freqlevel(self,
                        fmin: float,
                        fmax: float,
                        nF: int,
                        sparam: tuple[int, int],
                        upper_limit: float = None,
                        lower_limit: float = None,
                        weight: float = 1,
                        f_norm: float | PNorm = 4):
            """ Add a frequency level requirement to the optimiser.
            
            This method adds a frequency level requirement to the optimiser.
            
            Parameters
            ----------
            fmin : float
                The minimum frequency of the requirement.
            fmax : float
                The maximum frequency of the requirement.
            nF : int
                The number of frequencies to use in the requirement.
            sparam : tuple[int, int]
                The S-parameter to use in the requirement.
            upper_limit : float, optional
                The upper limit of the requirement. If not specified, the requirement is unbounded.
            lower_limit : float, optional
                The lower limit of the requirement. If not specified, the requirement is unbounded.
            weight : float, optional
                The weight of the requirement. If not specified, the weight is 1.
            f_norm : float | PNorm, optional
                The norm to use in the requirement. If not specified, the P2 norm is used.
            """
            req = FrequencyLevel(fmin, fmax, nF, sparam, upper_limit=upper_limit, lower_limit=lower_limit, weight=weight, f_norm=f_norm)
            self.add_requirement(req)

    def generate_objective(self, 
                           pnorm=2, 
                           initial: np.ndarray = None, 
                           differential_weighting: float = 0,
                           differential_weighting_exponent: float = 5) -> Callable:
        """ Generate the objective function for the optimiser.
        
        This method generates the objective function for the optimiser.
        
        Parameters
        ----------
        pnorm : int, optional
            The norm to use in the optimisation. If not specified, the P2 norm is used.
        initial : np.ndarray, optional
            The initial values of the optimisation variables. If not specified, the default values are used.
        differential_weighting : float, optional
            The differential weighting to use in the optimisation. If not specified, no differential weighting is used.
        differential_weighting_exponent : float, optional
            The exponent to use in the differential weighting. If not specified, the default value is used.
        
        Returns
        -------
        Callable
            The objective function for the optimiser.
        
        """
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
                return (np.mean(Ms**pnorm))**(1/pnorm) + differential_weighting*(max(Ms)-min(Ms))**differential_weighting_exponent
        else:
            def objective(coeffs):
                for p,c in zip(self.parameters, coeffs):
                    p.set(c)
                S = self.network.run(fs)
                subobj = [abs(req.eval(S))**pnorm for req in self.requirements]
                return (sum(subobj)/NR)**(1/pnorm)
               
        return objective
    



