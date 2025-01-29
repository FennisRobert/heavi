from __future__ import annotations
import numpy as np
from typing import Callable, Generator
from itertools import product

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

    @property
    def value(self) -> float:
        return self(self._eval_f)
    
    def scalar(self) -> float:
        """Returns the scalar value of the parameter"""
        return self(self._eval_f)
    
    def initialize(self) -> None:
        """Initializes the parameter"""
        pass

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Returns the value of the parameter at a given frequency"""
        return np.ones_like(f) * self._value
    
    def __repr__(self) -> str:
        """Returns a string representation of the parameter"""
        return f"SimParam({self._value})"
    
    def negative(self) -> SimParam:
        """Returns the negative of the parameter"""
        return Function(lambda f: -self(f))
    
    def inverse(self) -> SimParam:
        """Returns the inverse of the parameter"""
        return Function(lambda f: 1/self(f))
    
class Scalar(SimParam):
    """A class to represent a scalar value"""

    def __init__(self, value: float):
        self._value = value

    def __repr__(self) -> str:
        return f"SimValue({self._value})"
    
    def negative(self) -> Scalar:
        """Returns the negative of the scalar"""
        return Scalar(-self._value)
    
    def inverse(self) -> Scalar:
        """Returns the inverse of the scalar"""
        return Scalar(1/self._value)
    
class Negative(SimParam):
    """A class to represent the negative of a parameter"""

    def __init__(self, value: Scalar):
        self._value: Scalar = value

    def __repr__(self) -> str:
        return f"Negative({self._value})"
    
    def __call__(self, f: np.ndarray) -> np.ndarray:
        return -self._value(f)
    
    def negative(self) -> Scalar:
        """Returns the negative of the negative"""
        return self._value

class Inverse(SimParam):
    
    def __init__(self, value: Scalar):
        self._value: Scalar = value

    def __repr__(self) -> str:
        return f"Inverse({self._value})"
    
    def __call__(self, f: np.ndarray) -> np.ndarray:
        return 1/self._value(f)
    
    def inverse(self) -> Scalar:
        """Returns the inverse of the inverse"""
        return self._value
    
class Function(SimParam):

    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        """A class to represent a function of frequency.
        
        Parameters:
        -----------
        function : Callable[[np.ndarray], np.ndarray]
            The function of frequency.
        """
        self._function = function

    def __repr__(self) -> str:
        return f"Function({self._function})"
    
    def __call__(self, f: np.ndarray) -> np.ndarray:
        return self._function(f)
    
class Random(SimParam):

    def __init__(self, randomizer: Callable):
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

    def initialize(self):
        self._value = self._randomizer()

    def __repr__(self):
        return f"Gaussian({self._mean}, {self._std})"   
    
    def __call__(self, f):
        return self._value * np.ones_like(f)
    
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
    def __init__(self, values: np.ndarray):
        super().__init__()
        self._values = values
        self._index: int = 0
        self._value = Uninitialized()

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
        return self._value * np.ones_like(f)
    
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
    
    def iterate(self) -> Generator[tuple[tuple[int], tuple[float]], None, None]:
        '''An iterator that first compiles the total.'''
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
    
def parse_numeric(value: float | Scalar | Callable, inverse: bool = False) -> SimParam:
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
        return Function(value)
    elif isinstance(value, (int, float, complex)):
        if inverse:
            Scalar(1/value)
        return Scalar(value)
    else:
        raise ValueError(f"Invalid value type: {type(value)}")
    
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