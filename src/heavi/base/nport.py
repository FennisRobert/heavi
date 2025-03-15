from .component import BaseComponent, Node, SimParam, enforce_simparam, SimulationType, _TWO_MAT
from typing import List, Callable
import numpy as np

class NPortS(BaseComponent):
    supported_simulations: List[SimulationType] = [SimulationType.SP]
    
    def __init__(self, nodes: List[Node], gnd: Node, functions: List[List[Callable[[float], float]]], Z0: float):
        super().__init__()
        self.nodes = nodes
        self.nterminals = len(nodes)
        self.gnd = gnd
        self.functions = functions
        self.Z0: SimParam = Z0
        self.component_name = 'NPort(S)'
        self.unit = 'Ohm'
        self.value = self.Z0
    
    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([node.index for node in self.nodes] + [self.gnd.index,])
        return np.ix_(ids, ids)

    def SP_matrix_compiler(self):
        row, col = self.mat_slice
        N = self.nterminals

        def comp_function(f: float):
            nF = f.shape[0]
            S = np.array([[sp(f) for sp in row] for row in self.functions], dtype=np.complex128)
            Identity = np.repeat(np.eye(N)[:, :, np.newaxis], nF, axis=2).astype(np.complex128)
            Y = (1/self.Z0) * np.einsum('ijk,jlk->ilk', (Identity-S), np.stack([np.linalg.inv((Identity+S)[:, :, m]) for m in range(nF)],axis=2))
            Y2 = np.zeros((N+1,N+1,nF),dtype=np.complex128)
            Y2[:N,:N,:] = Y
            for i in range(N):
                Y2[i,N,:] = -np.sum(Y[i,:,:],axis=0)
                Y2[N,i,:] = -np.sum(Y[:,i,:],axis=0)
                Y2[N,N,:] += np.sum(Y[i,:,:],axis=0)
            return Y2
        
        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += comp_function(f)
            return matrix
        
        return compiler
    
class NPortY(BaseComponent):

    supported_simulations: List[SimulationType] = [SimulationType.SP]

    def __init__(self, nodes: List[Node], gnd: Node, functions: List[List[Callable[[float], float]]], Z0: float | SimParam):
        super().__init__()
        self.nodes = nodes
        self.nterminals = len(nodes) + 1
        self.nports = len(nodes)
        self.gnd = gnd
        self.functions = functions
        self.Z0: SimParam = enforce_simparam(Z0)
        self.component_name = 'NPort(Y)'
        self.unit = 'Ohm'
        self.value = self.Y0.value

    @property
    def mat_slice(self) -> tuple[np.ndarray, np.ndarray]:
        ids = np.array([node.index for node in self.nodes] + [self.gnd.index,])
        return np.ix_(ids, ids)

    def SP_matrix_compiler(self):
        row,col = self.mat_slice
        N = self.nterminals
        
        def comp_function(f: float):
            nF = f.shape[0]
            Y = np.array([[y(f) for y in row] for row in self.functions], dtype=np.complex128)
            Y2 = np.zeros((N+1,N+1,nF),dtype=np.complex128)
            Y2[:N,:N,:] = Y
            for i in range(N):
                Y2[i,N,:] = -np.sum(Y[i,:,:],axis=0)
                Y2[N,i,:] = -np.sum(Y[:,i,:],axis=0)
                Y2[N,N,:] += np.sum(Y[i,:,:],axis=0)
            return Y2
        
        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            matrix[row,col,:] += comp_function(f)
            return matrix
        
        return compiler
        
        
## NportS
# N = len(nodes)

# def comp_function(f: float):
#     nF = f.shape[0]
#     S = np.array([[sp(f) for sp in row] for row in Sparam], dtype=np.complex128)
#     Identity = np.repeat(np.eye(N)[:, :, np.newaxis], nF, axis=2).astype(np.complex128)
#     Y = (1/Z0) * np.einsum('ijk,jlk->ilk', (Identity-S), np.stack([np.linalg.inv((Identity+S)[:, :, m]) for m in range(nF)],axis=2))
#     Y2 = np.zeros((N+1,N+1,nF),dtype=np.complex128)
#     Y2[:N,:N,:] = Y
#     for i in range(N):
#         Y2[i,N,:] = -np.sum(Y[i,:,:],axis=0)
#         Y2[N,i,:] = -np.sum(Y[:,i,:],axis=0)
#         Y2[N,N,:] += np.sum(Y[i,:,:],axis=0)
#     return Y2
# component = Component(nodes + [gnd, ],[ComponentFunction(comp_function),], Z0).set_metadata(value=Z0, **Components.NPORT)
# self.components.append(component)

## NPORTY
