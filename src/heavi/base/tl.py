from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT
from typing import List, Callable
import numpy as np

class TransmissionLine(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.SP,]
    
    def __init__(self, node1: Node, node2: Node, gnd: Node, Z0: float | SimParam, length: float, beta: Callable | SimParam):
        super().__init__()
        self.nodes = [node1, node2, gnd]
        self.gnd = gnd
        self.nterminals = 4
        self.Z0: SimParam = enforce_simparam(Z0)
        self.beta: SimParam = enforce_simparam(beta)
        self.length: float = length
        self.component_name = 'TransmissionLine'
        self.unit = 'Ohm'
        self.value = self.Z0.value
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([self.nodes[0].index, self.nodes[1].index, self.gnd.index])
        return np.ix_(ids, ids)
    
    def SP_matrix_compiler(self):
        beta = self.beta
        length = self.length
        Z0 = self.Z0
        
        def a11(f):
            return np.cosh(1j*beta(f)*length)
        def a12(f):
            return Z0(f)*np.sinh(1j*beta(f)*length)
        def a21(f):
            return 1/Z0(f)*np.sinh(1j*beta(f)*length)
        def a22(f):
            return np.cosh(1j*beta(f)*length)
        
        def y11(f):
            return a22(f)/a12(f)
        def y12(f):
            return -((a11(f)*a22(f))-(a12(f)*a21(f)))/a12(f)
        def y21(f):
            return -1/a12(f)
        def y22(f):
            return a11(f)/a12(f)
        
        functions = [[y11, y12], [y21, y22]]
        N = 2
        def comp_function(f: float):
            nF = f.shape[0]
            Y = np.array([[y(f) for y in row] for row in functions], dtype=np.complex128)
            Y2 = np.zeros((N+1,N+1,nF),dtype=np.complex128)
            Y2[:N,:N,:] = Y
            for i in range(2):
                #Y2[i,i,:] += np.sum(Y[i,:,:],axis=0)
                Y2[i,N,:] = -np.sum(Y[i,:,:],axis=0)
                Y2[N,i,:] = -np.sum(Y[:,i,:],axis=0)
                Y2[N,N,:] += np.sum(Y[i,:,:],axis=0)
            return Y2
        
        row, col = self.mat_slice
        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += comp_function(f)
            return matrix
        
        return compiler