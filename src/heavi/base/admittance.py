from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT
from typing import List, Callable
import numpy as np
from .numeric import stack, cstack



class Admittance(BaseComponent):
    
    supported_simulations: List[SimulationType] = [SimulationType.AC,
                                                   SimulationType.DC, 
                                                   SimulationType.SP,]
    
    def __init__(self, node1: Node, node2: Node, admittance: float | SimParam):
        '''
        Adds a generic admittance component to the circuit. The admittance can have a float
        or frequency dependent value through the SimParameter class.

        Returns:
            Admittance: The admittance component 
        '''
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        self.Y: SimParam = enforce_simparam(admittance)
        self.component_name = 'Admittance'
        self.unit = 'S'
        self.value = self.Y(self.PRINT_FREQ)
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.nodes[0].index, self.nodes[1].index])
        return np.ix_(ids, ids)
    
    def generic_matrix_compiler(self) -> Callable:
        row,col = self.mat_slice

        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += cstack(_TWO_MAT, self.Y(f), 0*_TWO_MAT)
            return matrix
        
        return compiler
    
    def SP_matrix_compiler(self):
        return self.generic_matrix_compiler()
    
    def DC_matrix_compiler(self):
        row,col = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[row,col] += self.Y(0.0) * _TWO_MAT
            return matrix
        
        return compiler
    
    def AC_matrix_compiler(self):
        return self.generic_matrix_compiler()

