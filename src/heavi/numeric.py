from __future__ import annotations
import numpy as np
from typing import Callable, Generator
from itertools import product
from .sparam import NDSparameters
import inspect
from functools import wraps

def _num_args(func):
    sig = inspect.signature(func)
    params = sig.parameters
    return len(params)



TEN_POWERS = {
    -12: "p",
    -9: "n",
    -6: "u",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "T",
    12: "P",
}


def _get_power(number: float):
    tp = np.log10(np.abs(number))
    v = np.floor(tp / 3) * 3
    v = min(12, max(-12, v))
    v2 = number / (10**v)
    return v2, v

def format_value_units(value: float, unit: str) -> str:
    """ Formats a value with units for display. """
    
    v, p = _get_power(value)
    return f"{v:.2f} {TEN_POWERS[p]}{unit}"

class Uninitialized:
    """A class to represent an uninitialized value"""
    def __init__(self):
        pass

    def __repr__(self):
        return "Uninitialized"
    
    def __call__(self, f):
        raise ValueError("Uninitialized value")
    
    def __mul__(self, other):
        raise ValueError("Uninitialized value")
    
    def __add__(self, other):
        raise ValueError("Uninitialized value")
    
    def __sub__(self, other):
        raise ValueError("Uninitialized value") 
    

class SimParam:
    _eval_f = 1e9

    def __init__(self, unit=''):
        self.unit = ''
        self._inf = np.inf
        

    @property
    def value(self) -> float:
        return self(0)
    
    def scalar(self) -> float:
        """Returns the scalar value of the parameter"""
        return self.value
    
    def initialize(self) -> None:
        """Initializes the parameter"""
        pass

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Returns the value of the parameter at a given frequency"""
        return np.nan_to_num(np.ones_like(f) * self._value, posinf=self._inf, neginf=-self._inf)
    
    def __repr__(self) -> str:
        """Returns a string representation of the parameter"""
        return f"SimParam({self._value})"
    
    def negative(self) -> SimParam:
        """Returns the negative of the parameter"""
        return Function(lambda f: -self.sp(f))
    
    def inverse(self) -> SimParam:
        """Returns the inverse of the parameter"""
        return Function(lambda f: 1/self.sp(f))
    
class Scalar(SimParam):
    """A class to represent a scalar value"""

    def __init__(self, value: float, unit: str = '', inf: float = np.inf):
        self._value = value
        self._inf = inf
        self.unit = unit
    
    def __repr__(self) -> str:
        return f"SimValue({self._value})"
    
    def negative(self) -> SimParam:
        """Returns the negative of the scalar"""
        return Scalar(-self._value)
    
    def inverse(self) -> SimParam:
        """Returns the inverse of the scalar"""
        return Scalar(1/self._value)

    
class Negative(SimParam):
    """A class to represent the negative of a parameter"""

    def __init__(self, simparam: SimParam, inf: float = np.inf):
        self._simparam: SimParam = simparam
        self.unit = simparam.unit
        self._inf = inf

    def __repr__(self) -> str:
        return f"Negative({self._simparam})"
    
    def __call__(self, f: np.ndarray) -> np.ndarray:
        return np.nan_to_num(-self._simparam(f), posinf=self._inf, neginf=-self._inf)
    
    def negative(self) -> SimParam:
        """Returns the negative of the negative"""
        return self._simparam

class Inverse(SimParam):
    
    def __init__(self, simparam: SimParam, inf: float = np.inf):
        self._simparam: SimParam = simparam
        self.unit = simparam.unit
        self._inf = inf

    def __repr__(self) -> str:
        return f"Inverse({self._simparam})"
    
    def __call__(self, f: np.ndarray) -> np.ndarray:
        return np.nan_to_num(1/self._simparam(f), posinf=self._inf, neginf=-self._inf)
    
    def inverse(self) -> SimParam:
        """Returns the inverse of the inverse"""
        return self._simparam
    
class Function(SimParam):

    def __init__(self, 
                 function: Callable[[np.ndarray], np.ndarray], 
                 unit: str ='',
                 inf: float = np.inf):
        """A class to represent a function of frequency.
        
        Parameters:
        -----------
        function : Callable[[np.ndarray], np.ndarray]
            The function of frequency.
        """
        self._func = function
        self._inf = inf
        self.unit: str = unit

    def __call__(self, f: np.ndarray) -> np.ndarray:
        return np.nan_to_num(self._func(f), posinf=self._inf, neginf=-self._inf)

    def __repr__(self) -> str:
        return f"Function({self._func})"
    
    
class Random(SimParam):

    def __init__(self, randomizer: Callable, unit: str = '', inf: float = np.inf):
        """A class to represent a random value.
        
        Parameters:
        -----------
        randomizer : Callable
            A function that returns a random value.
        """
        super().__init__()
        self._randomizer = randomizer
        self._value = Uninitialized()
        self._mean = None
        self._std = None
        self._inf = inf
        self.unit = unit

    def initialize(self):
        self._value = self._randomizer()

    def __repr__(self):
        return f"Gaussian({self._mean}, {self._std})"   
    
    def negative(self) -> SimParam:
        return Negative(self)
    
    def inverse(self) -> SimParam:
        return Inverse(self)
    
class Param(SimParam):
    """ A class to represent a parameter that is swept over a range of values.

    Parameters:
    -----------
    values : np.ndarray
        An array of values to sweep over
    """
    def __init__(self, values: np.ndarray, unit: str = '', inf: float = np.inf):
        super().__init__()
        self._values = values
        self._index: int = 0
        self._value = Uninitialized()
        self._inf = inf
        self.unit: str = unit

    @staticmethod
    def lin(start: float, stop: float, Nsteps: int) -> Param:
        """Creates a linearly spaced parameter sweep."""
        return Param(np.linspace(start,stop,Nsteps))
    
    @staticmethod
    def range(start: float, stop: float, step: float, *args, **kwargs) -> Param:
        """Creates a range of values with a given step size."""
        return Param(np.arange(start, stop, step, *args, **kwargs))

    def __len__(self):
        return len(self._values)
    
    def __repr__(self):
        ## shortened list of values (start and end only)
        return f"Param({self._values[0]}, ..., {self._values[-1]})"
    
    def __call__(self, f):
        return np.nan_to_num(self._value * np.ones_like(f), posinf=self._inf, neginf=-self._inf)
    
    def set_index(self, index: int):
        """Sets the index of the parameter.
        
        Parameters:
        -----------
        index : int
            The index of the parameter.
        """

        self._index = index
    
    def initialize(self):
        self._value = self._values[self._index]
    
    def negative(self) -> SimParam:
        return Negative(self)
    
    def inverse(self) -> SimParam:
        return Inverse(self)
    
class ParameterSweep:

    def __init__(self):
        self.sweep_dimensions: list[tuple[Param]] = []
        self.index_series: list[tuple[int]] = []
        self._index: int = 0
        self._param_buffer: list = []
        self._S_data: list[np.ndarray] = []
        self._current_index: tuple[int] = None
        self._mdim_data: NDSparameters = None
        self._fdata: np.ndarray = None
    
    def lin(self, start: float, stop: float) -> ParameterSweep:
        """Adds a linear sweep of values to the parameter sweep.
        
        Parameters:
        -----------
        start : float
            The start value of the sweep.
        stop : float
            The stop value of the sweep.
        
        Returns:
        --------
        ParameterSweep
            The parameter sweep object.
        """

        self._param_buffer.append((start,None,stop))
        return self
    
    def step(self, start: float, stepsize: float) -> ParameterSweep:
        """Adds a stepped sweep of values to the parameter sweep.
        
        Parameters:
        -----------
        start : float
            The start value of the sweep.
        stepsize : float
            The step size of the sweep.
        
        Returns:
        --------
        ParameterSweep
            The parameter sweep object.
        """
        self._param_buffer.append((start,stepsize,None))

    def add(self, N: int) -> tuple[Param]:
        """Adds a dimension to the parameter sweep.
        
        Parameters:
        -----------
        N : int
            The number of steps in the dimension.
        
        Returns:
        --------
        tuple[Param]
            A tuple of Param objects."""
        params = []
        for start, step, stop in self._param_buffer:
            if step is None:
                params.append(Param.lin(start,stop,N))
            elif stop is None:
                params.append(Param.lin(start,start+step*N,N))
        self.add_dimension(*params)
        self._param_buffer = []
        return params
    
    def submit(self, S: np.ndarray, fs: np.ndarray) -> ParameterSweep:
        """Submits a set of S-parameters to the parameter sweep.
        
        Parameters:
        -----------
        S : np.ndarray
            The S-parameters to submit.
        """
        self._S_data.append((self._current_index, S))
        self._fdata = fs
        return self

    @property
    def S(self) -> NDSparameters:
        """Returns the S-parameters of the parameter sweep."""
        if self._mdim_data is None:
            self._mdim_data = self.generate_mdim()
        return self._mdim_data
    

    def generate_mdim(self) -> NDSparameters:
        """Generates a multi-dimensional array of S-parameters."""
        shape = [len(dimension[0]) for dimension in self.sweep_dimensions]
        s_proto = self._S_data[0][1]
        nports = s_proto.shape[0]
        ports = np.arange(1, nports+1)
        S = np.zeros(shape + list(self._S_data[0][1].shape), dtype=np.complex128)
        for index, s in self._S_data:
            S[index] = s._S
        NDS = NDSparameters(S, [np.array([dim._values for dim in dimension]) for dimension in self.sweep_dimensions] + [ports, ports, self._fdata])
        return NDS

    def iterate(self) -> Generator[tuple[tuple[int], tuple[float]], None, None]:
        '''An iterator that first compiles the total.'''
        self._S_data: list[np.ndarray] = []

        # Make a list of all dimensional index tuples as the product of the lengths of each dimension
        total = 1
        for dimension in self.sweep_dimensions:
            total *= len(dimension)

        # make a check for total above 10,000
        if total > 10000:
            raise ValueError(f"Total iterations ({total}) is above 10,000, are you sure you want to continue?")
        
        # Get all the length of the dimensions of the parameter sweep
        lengths = [len(dimension[0]) for dimension in self.sweep_dimensions]

        # make a list of indices like [(0,0,0),(1,0,0),(2,0,),...,(N,M,K)] using itertools
        indices = list(product(*[range(length) for length in lengths]))
        
        for ixs in indices:
            paramlist = []
            self._current_index: tuple[int] = ixs
            # set the index of each dimensional Param object
            for i, params in zip(ixs, self.sweep_dimensions):
                for param in params:
                    param.set_index(i)
                    param.initialize()
                    paramlist.append(param._value)
            yield ixs, tuple(paramlist)

    def add_dimension(self, *params: tuple[Param]):
        self.sweep_dimensions.append(params)

class MonteCarlo:
    """A class to represent a Monte Carlo simulation."""
    def __init__(self):
        self._random_numbers: list[Random] = []

    def gaussian(self, mean: float, std: float) -> Random:
        """Adds a Gaussian random number to the Monte Carlo simulation.
        
        Parameters:
        -----------
        mean : float
            The mean of the Gaussian distribution.
        std : float
            The standard deviation of the Gaussian distribution.
        
        Returns:
        --------
        Random
            A Random parameter object.
        """
        random = Random(lambda: np.random.normal(mean, std))
        random._mean = mean
        random._std = std
        self._random_numbers.append(random)
        return random
    
    def uniform(self, low: float, high: float) -> Random:
        """Adds a uniform random number to the Monte Carlo simulation.
        
        Parameters:
        -----------
        low : float
            The lower bound of the uniform distribution.
        high : float
            The upper bound of the uniform distribution.
        
        Returns:
        --------
        Random
            A Random parameter object.
        """
        random = Random(lambda: np.random.uniform(low, high))
        self._random_numbers.append(random)
        return random
    
    def iterate(self, N: int) -> Generator[int, None, None]:
        for i in range(N):
            for random in self._random_numbers:
                random.initialize()
            yield i


def enforce_simparam(value: float | SimParam | Callable, inverse: bool = False, unit: str = '') -> SimParam:
    """Parses a numeric value to a SimParam object.

    Parameters:
    -----------
    value : float | Scalar | Callable
        The value to parse.
    inverse : bool
        Whether to return the inverse of the value.
    
    Returns:
    --------
    SimParam
        The SimParam object.
    """
    if isinstance(value, SimParam):
        if inverse:
            return value.inverse()
        return value
    elif isinstance(value, Callable):
        if inverse:
            return Function(lambda f: 1/value(f))
        return Function(value, unit=unit)
    elif isinstance(value, (int, float, complex)):
        if inverse:
            return Scalar(1/value, unit=unit)
        return Scalar(value, unit=unit)
    else:
        raise ValueError(f"Invalid value type: {type(value)}")
    

def ensure_simparam(func):
    # Get the function signature
    sig = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Convert arguments to float if their type hint is complex or float
        for name, value in bound_args.arguments.items():
            param = sig.parameters[name]
            if param.annotation in (float, SimParam, float | SimParam, SimParam | float):
                bound_args.arguments[name] = enforce_simparam(value)

        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper

def set_print_frequency(frequency: float) -> None:
    """Sets the frequency at which the simulation parameters are evaluated for printing.
    
    Parameters:
    -----------
    frequency : float
        The frequency at which the simulation parameters are evaluated.
    """
    # check if frequency is a float with a valid value
    if not isinstance(frequency, (int, float)):
        raise ValueError("Frequency must be a float")
    
    # check if it is greater than 0 and less than the upper frequency of the optical region
    if 0 < frequency < 1e15:
        raise ValueError("Frequency must be greater than 0 and less than 1e15 Hz")
    
    SimParam._eval_f = frequency