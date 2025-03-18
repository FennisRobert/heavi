from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT_C, _TWO_MAT
from typing import List, Callable
import numpy as np
from .numeric import stack, cstack

class Capacitor(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.AC,
                                                   SimulationType.DC, 
                                                   SimulationType.SP,
                                                   SimulationType.TRANS]
    
    def __init__(self, node1: Node, node2: Node, capacitance: float):
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        self.C: float = capacitance
        
        self.component_name = 'Capacitor'
        self.unit = 'F'
        self.value = capacitance
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.nodes[0].index, self.nodes[1].index])
        return np.ix_(ids, ids)
    
    def SP_matrix_compiler(self):
        row,col = self.mat_slice

        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += cstack(_TWO_MAT_C, 1j*2*np.pi*f*self.C ,0*_TWO_MAT_C)
            return matrix
        
        return compiler
    
    def DC_matrix_compiler(self):
        slc = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[slc] += 0 * _TWO_MAT
            return matrix
        
        return compiler
    
    def AC_matrix_compiler(self):
        return self.SP_matrix_compiler()
    