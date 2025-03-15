########################################################################################
##
##    Resistor component Definition
##    This file contains the definition of the Resistor component.
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################


from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT
from typing import List, Callable
import numpy as np


#  __             __   __  
# /  ` |     /\  /__` /__` 
# \__, |___ /~~\ .__/ .__/ 
# -------------------------------------------


class Resistor(BaseComponent):
    '''
    """The Resistor component models a simple resistor component without paracitic properties.

    Returns:
        Resistor: Resistor component
    """    '''
    supported_simulations: List[SimulationType] = [SimulationType.AC,
                                                   SimulationType.DC, 
                                                   SimulationType.SP,
                                                   SimulationType.TRANS]
    
    def __init__(self, node1: Node, node2: Node, resistance: float):
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        self.G: float = 1/resistance
        self.R: float = resistance
        
        self.component_name = 'Resistor'
        self.unit = 'Ohm'
        self.value = resistance
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.nodes[0].index, self.nodes[1].index])
        return np.ix_(ids, ids)
    
    def generic_matrix_compiler(self) -> Callable:
        slc = self.mat_slice

        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[slc] += self.G * _TWO_MAT
            return matrix
        
        return compiler
    
    def SP_matrix_compiler(self):
        return self.generic_matrix_compiler()
    
    def DC_matrix_compiler(self):
        slc = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[slc] += self.G * _TWO_MAT
            return matrix
        
        return compiler
    
    def AC_matrix_compiler(self):
        return self.generic_matrix_compiler()
    
    def TRANS_matrix_compiler(self):
        slc = self.mat_slice

        def compiler(matrix: np.ndarray) -> np.ndarray:
            matrix[slc] += self.G * _TWO_MAT
            return matrix
        
        return compiler
    