from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT_C, _TWO_MAT
from typing import List, Callable
import numpy as np
from .numeric import cstack, stack

class Impedance(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.AC,
                                                   SimulationType.DC, 
                                                   SimulationType.SP,]
    
    def __init__(self, node1: Node, node2: Node, impedance: float | SimParam):
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        self.Z: SimParam = enforce_simparam(impedance)
        self.component_name = 'Impedance'
        self.unit = 'Ohm'
        self.value = self.Z(self.PRINT_FREQ)
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.nodes[0].index, self.nodes[1].index])
        return np.ix_(ids, ids)
    
    def generic_matrix_compiler(self) -> Callable:
        row,col = self.mat_slice

        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += cstack(_TWO_MAT_C, 1/self.Z(f), 0*_TWO_MAT_C)
            return matrix
        
        return compiler
    
    def SP_matrix_compiler(self):
        return self.generic_matrix_compiler()
    
    def DC_matrix_compiler(self):
        row,col = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[row,col] += self.Z(0.0) * _TWO_MAT
            return matrix
        
        return compiler
    
    def AC_matrix_compiler(self):
        return self.generic_matrix_compiler()