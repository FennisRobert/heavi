from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT
from .numeric import stack, cstack
from typing import List, Callable
import numpy as np

class Inductor(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.AC,
                                                   SimulationType.DC, 
                                                   SimulationType.SP,
                                                   SimulationType.TRANS]
    
    TRANS_sources = 1
    def __init__(self, node1: Node, node2: Node, inductance: float):
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        self.L: float = inductance
        
        self.component_name = 'Inductor'
        self.unit = 'H'
        self.value = inductance
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.nodes[0].index, self.nodes[1].index])
        return np.ix_(ids, ids)
    
    def SP_matrix_compiler(self):
        row,col = self.mat_slice

        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += cstack(_TWO_MAT, 1/(1j*2*np.pi*f*self.L),0*_TWO_MAT)
            return matrix
        
        return compiler
    
    def DC_matrix_compiler(self):
        slc = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[slc] += 1e9 * _TWO_MAT
            return matrix
        
        return compiler
    
    def AC_matrix_compiler(self):
        return self.SP_matrix_compiler()
    