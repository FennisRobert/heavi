import numpy as np
from typing import Callable
from ..rfcircuit import _stack, _TWO_MAT, ComponentFunction, NonLinearComponent
from .libgen import BaseComponent
from scipy.optimize import fsolve

class CurrentFunction(BaseComponent):
    def __init__(self, function: Callable[[float], float]):
        super().__init__()
        self.n_nodes = 2
        self.function: Callable[[float], float] = function
        self.voltage: float = 0.5
        self.eps: float = 1e-15

    def get_voltage(self, V: np.ndarray) -> float:
        i1 = self.node(1)
        i2 = self.node(2)
        print(f'Measured voltage: {V[i1.index] - V[i2.index]}')
        self.voltage = V[i1.index] - V[i2.index]

    def _fG0(self, f: float) -> np.ndarray:
        i2 = self.function(self.voltage+self.eps)
        i1 = self.function(self.voltage-self.eps)
        return (i2-i1)/(2*self.eps) * np.ones_like(f)

    def _g_stencil(self, f: float) -> np.ndarray:
        print(f'Voltage: {self.voltage}')
        print(f'Current: {self.function(self.voltage)}')
        print(f'fG0: {self._fG0(f)}')
        return self._fG0(f) * _stack(_TWO_MAT, f)
    
    def _v_stencil(self, f: float) -> np.ndarray:
        print(f'Voltage: {self.voltage}')
        print(f'I_diode: {self.function(self.voltage)}')
        print(f'I_deriv: {self._fG0(f)*self.voltage}')
        vtencil =  _stack(np.array([[-1],[1]]),f)*(self.function(self.voltage) - self._fG0(f)*self.voltage)
        print(f'vs: {vtencil}')
        return vtencil

    def __on_connect__(self):
        node1 = self.node(1)
        node2 = self.node(2)

        functionlist = [ComponentFunction([[node1, node2],[node1, node2]], self._g_stencil),]
        sourcefunlist = [ComponentFunction([[node1, node2],], self._v_stencil),]
        admittance_component = NonLinearComponent([node1, node2], 
                                                  functions=functionlist, 
                                         source_functions=sourcefunlist,
                                         v_function= self.get_voltage,
                                         component_value=0).set_metadata(name='NonLinearFunction')
        self.network.components.append(admittance_component)


class Diode(CurrentFunction):

    def __init__(self, Vt: float=0.0258, N=1.3497, I0: float = 1e-12, Rs = 0.13668):
        
        def diode_current(voltage):
            def equation(current):
                return current - I0 * (np.exp((voltage.real - current * Rs) / (N * Vt)) - 1)
            
            # Solve for I numerically (implicit equation)
            I_sol = fsolve(equation, 0.1)  # Initial guess: 0
            return I_sol

        
        super().__init__(diode_current)
    