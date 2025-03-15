import numpy as np
from typing import List, Callable, Tuple, Literal
from ..node import Node, MatSource
from ..numeric import SimParam, enforce_simparam
from enum import Enum

_TWO_MAT = np.array([[1.0,-1.0],[-1.0,1.0]])
_TWO_MAT_C = np.array([[1.0,-1.0],[-1.0,1.0]]).astype(np.complex128)

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

class SimulationType(Enum):
    AC = 1
    DC = 2
    TRANS = 3
    SP = 4

def _get_power(number: float):
    ''' Splits a float into a base value and the power of 10 in groups of 1000.
    
    >>> _get_power(4532)
    (4.532, 3)
    '''
    N_thousands = np.floor(np.log10(np.abs(number)) / 3) * 3
    N_thousands = min(12, max(-12, N_thousands))
    base_number = number / (10**N_thousands)
    return base_number, N_thousands

def _format_value_units(value: float, unit: str) -> str:
    """ Formats a value with units for display. 
    
    >>> format_value_units(4532, 'Hz')
    '4.53 kHz'
    """
    
    v, p = _get_power(value)
    return f"{v:.2f} {TEN_POWERS[p]}{unit}"

class BaseComponent:

    supported_simulations: List[SimulationType] = []
    
    PRINT_FREQ: float = 1e0

    AC_sources: int = 0
    DC_sources: int = 0
    TRANS_sources: int = 0
    SP_sources: int = 0

    AC_INTNODES: int = 0
    DC_INTNODES: int = 0
    TRANS_INTNODES: int = 0
    SP_INTNODES: int = 0

    _is_linear: bool = True

    def __init__(self):
        self.nodes: List[Node] = []
        self.nterminals: int = 0
        self.sources: List[MatSource] = []
        self.internal_nodes: List[Node] = []

        self.component_name: str = 'BaseComponent'
        self.unit: str = ''
        self.value: float = 'Undefined'

        self.metadata: dict[str,str] = {}

    def __repr__(self):
        return f'{self.component_name}[{_format_value_units(self.value, self.unit)}]'

    def __str__(self):
        return self.__repr__()
    
    def summary(self) -> str:
        ''' Returns a multiline string summarizing component data. '''
        base = dict(name=self.component_name, value=self.value, unit=self.unit)
        base.update(self.metadata)

        lines = [f'{base["name"]} ({_format_value_units(base["value"], base["unit"])})']
        del base["name"]
        del base["value"]
        del base["unit"]
        if base:
            lines.append('-'*len(lines[0]))
            for key, value in base.items():
                lines.append(f'    {key}: {value}')
        return '\n'.join(lines)
    
    def set_metadata(self, **kwargs):
        self.metadata.update(kwargs)
        
    @property
    def all_nodes(self) -> List[Node]:
        return self.nodes + self.internal_nodes
    
    def is_linear(self, simulation_type: SimulationType) -> bool:
        return self._is_linear
    
    def n_requested_sources(self, simulation_type: SimulationType) -> int:
        if simulation_type == SimulationType.AC:
            return self.AC_sources
        if simulation_type == SimulationType.DC:
            return self.DC_sources
        if simulation_type == SimulationType.TRANS:
            return self.TRANS_sources
        if simulation_type == SimulationType.SP:
            return self.SP_sources
    
    def n_requested_internal_nodes(self, simulation_type: SimulationType) -> int:
        if simulation_type == SimulationType.AC:
            return self.AC_INTNODES
        if simulation_type == SimulationType.DC:
            return self.DC_INTNODES
        if simulation_type == SimulationType.TRANS:
            return self.TRANS_INTNODES
        if simulation_type == SimulationType.SP:
            return self.SP_INTNODES
    
    def indices(self) -> List[int]:
        return [node.index for node in self.nodes] + [source.index for source in self.sources]
    
    def set_sources(self, sources: List[MatSource]):
        self.sources = sources

    def set_internal_nodes(self, nodes: List[Node]):
        self.internal_nodes = nodes

    @property
    def mat_slice(self) -> Tuple[np.ndarray, np.ndarray]:
        ids = np.array(self.indices())
        return np.ix_(ids,ids)
    
    @property
    def vec_slice(self) -> np.ndarray:
        return np.ix_(self.indices())

    def AC_matrix_compiler(self) -> Callable:
        return None
    
    def AC_source_compiler(self) -> Callable:
        return None
    
    def DC_matrix_compiler(self) -> Callable:
        return None
    
    def DC_source_compiler(self) -> Callable:
        return None
    
    def TRANS_matrix_compiler(self) -> Callable:
        return None
    
    def TRANS_source_compiler(self) -> Callable:
        return None
    
    def SP_matrix_compiler(self) -> Callable:
        return None
    
    def SP_source_compiler(self) -> Callable:
        return None
    
