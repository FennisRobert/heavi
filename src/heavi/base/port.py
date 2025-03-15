from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT, _TWO_MAT_C
from typing import List, Callable
import numpy as np
from .numeric import stack, cstack

class Port(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.SP,SimulationType.DC,SimulationType.AC]
    SP_sources = 1
    def __init__(self, index: int, sig_node: Node, int_node: Node, gnd_node: Node, Z0: float | SimParam):
        super().__init__()
        self.nodes = [sig_node, int_node, gnd_node]
        self.index: int = index
        self.sig_node: Node = sig_node
        self.int_node: Node = int_node
        self.gnd_node: Node = gnd_node
        self.nterminals = 2
        self.Z0: SimParam = enforce_simparam(Z0)
        self.component_name = 'Port'
        self.unit = 'Ohm'
        self.value = self.Z0.value
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.sig_node.index, self.int_node.index, self.gnd_node.index, self.sources[0].index])
        return np.ix_(ids, ids)
    
    def SP_matrix_compiler(self):
        row, col = self.mat_slice
        TEMPLATE = np.array([[ 1,-1, 0, 0], 
                             [-1, 1, 0, 0],
                             [ 0, 0, 0, 0],
                             [ 0, 0, 0, 0]])
        TEMPLATE_SOURCE = np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0,-1],
                                    [0, 1,-1, 0]])

        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += stack(TEMPLATE, 1/self.Z0(f), TEMPLATE_SOURCE)
            return matrix
        
        return compiler
    
    def DC_matrix_compiler(self):
        slc = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[slc] += 1/self.Z0.value * _TWO_MAT
            return matrix
        
        return compiler
    
    def AC_matrix_compiler(self):
        slc = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[slc] += cstack(_TWO_MAT_C, 1/self.Z0.value, 0*_TWO_MAT_C)
            return matrix
        
        return compiler