########################################################################################
##
##    Default Source Components
##    These components are used to define the behavior of voltage sources in the circuit
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################

#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------


from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT
from typing import List, Callable
import numpy as np


#  __             __   __   ___  __  
# /  ` |     /\  /__` /__` |__  /__` 
# \__, |___ /~~\ .__/ .__/ |___ .__/ 
# -------------------------------------------


class VoltageDC(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.AC,
                                                   SimulationType.DC,
                                                   SimulationType.SP,]
    
    DC_sources = 1
    
    def __init__(self, node1: Node, node2: Node, voltage: float, r_series: float | SimParam = 0.001):
        """Creates a DC voltage source component

        Args:
            node1 (Node): The positive terminal of the voltage source
            node2 (Node): The negative terminal of the voltage source
            voltage (float): The voltage of the source
            r_series (float | SimParam, optional): The series impedance. Defaults to 0.001.
        """        
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        self.V: float = voltage
        self.r_series: SimParam = enforce_simparam(r_series)
        self.component_name = 'DC Voltage Source'
        self.unit = 'V'
        self.value = self.V
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array(self.indices())
        return np.ix_(ids, ids)
    
    @property
    def vec_slice(self) -> np.ndarray:
        return np.array(self.indices())
    
    @property
    def index(self) -> int:
        return self.sources[0].index
    
    def DC_matrix_compiler(self):
        
        MAT_TEMPLATE = np.array([[0, 0, -1],
                             [0, 0, 1],
                             [-1, 1, 0]])
        slc = self.mat_slice

        def compiler(matrix: np.ndarray):
            matrix[slc] += MAT_TEMPLATE
            return matrix
        
        return compiler
    
    def DC_source_compiler(self):
        voltage = self.V
        index = self.sources[0].index

        def compiler(vector: np.ndarray):
            vector[index] = voltage
            return vector
        return compiler
    
    def SP_matrix_compiler(self):
        slc = self.mat_slice

        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[slc] += self.r_series(f) * _TWO_MAT
            return matrix
        
        return compiler
    
    def AC_matrix_compiler(self):
        return self.SP_matrix_compiler()
    

class VoltageFN(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.TRANS,]
    
    TRANS_sources = 1
    
    def __init__(self, node1: Node, node2: Node, voltage_function: float, r_series: float | SimParam = 0.001):
        """Creates a transient voltage source component"

        Args:
            node1 (Node): The positive terminal of the voltage source
            node2 (Node): The negative terminal of the voltage source
            voltage_function (float): The voltage function of the source
            r_series (float | SimParam, optional): The series impedance. Defaults to 0.001.
        
        """
        super().__init__()
        self.nodes = [node1, node2]
        self.nterminals = 2
        self.vfun: Callable = voltage_function
        self.r_series: SimParam = enforce_simparam(r_series)
        self.component_name = 'Transient Voltage Source'
        self.unit = 'V'
        self.value = 0
    
    @property
    def index(self) -> int:
        return self.sources[0].index
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array(self.indices())
        return np.ix_(ids, ids)
    
    @property
    def vec_slice(self) -> np.ndarray:
        return np.array(self.indices())
    
    def generate_timeseries(self, t: np.ndarray) -> np.ndarray:
        return self.vfun(t)
    
    def TRANS_matrix_compiler(self):
        
        MAT_TEMPLATE = np.array([[0, 0, -1],
                             [0, 0, 1],
                             [-1, 1, 0]])
        slc = self.mat_slice

        def compiler(matrix: np.ndarray):
            matrix[slc] += MAT_TEMPLATE
            return matrix
        
        return compiler
    
    def SP_matrix_compiler(self):
        slc = self.mat_slice
        # EFFECTIVE SHORT CIRCUIT
        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[slc] += self.r_series(f) * _TWO_MAT
            return matrix
        
        return compiler
    
    def DC_matrix_compiler(self):
        return self.SP_matrix_compiler()
    
    def AC_matrix_compiler(self):
        return self.SP_matrix_compiler()