from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT
from typing import List, Callable
from scipy.optimize import fsolve
from loguru import logger
import numpy as np

class Diode(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.DC, 
                                                   SimulationType.TRANS]
    
    _is_linear = False

    def __init__(self, node1: Node, node2: Node, Vt: float=0.0258, N=1.3497, I0: float = 1.2e-15, Rs = 0.13668):
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        
        self.Vt = Vt
        self.N = N
        self.I0 = I0
        self.Rs = Rs
        
        self.component_name = 'Diode'
        self.unit = 'V'
        self.value = self.Vt
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.nodes[0].index, self.nodes[1].index])
        return np.ix_(ids, ids)
    
    
    def IVData(self, Vmin: float = -100, Vmax: float = 5, Nsteps: int = 101):
        Vs = np.tan(np.linspace(np.arctan(Vmin), np.arctan(Vmax), Nsteps))
        
        Is = self.I0 * (np.exp(Vs/(self.N*self.Vt))-1)

        return Is, Vs

