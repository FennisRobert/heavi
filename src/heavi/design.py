from .rfcircuit import Network, Node, Z0_VSWR
import numpy as np
from typing import Tuple, List
from functools import reduce


def unbalanced_splitter(N: Network, node: Node, Z0: float, f0: float, er: float = 1) -> List[Node]:
    vg = 299792458/np.sqrt(er)
    wl = vg / f0

    n1 = N.node()
    n2 = N.node()
    Z02 = Z0*np.sqrt(2)
    N.transmissionline(N.gnd, node, n1, Z02, er, wl/4)
    N.transmissionline(N.gnd, node, n2, Z02, er, wl/4)

    return [n1, n2]

def balanced_splitter(N: Network, node: Node, Z0: float, f0: float, er: float = 1) -> List[Node]:
    vg = 299792458/np.sqrt(er)
    wl = vg / f0

    n1 = N.node()
    n2 = N.node()
    Z02 = Z0*np.sqrt(2)
    N.transmissionline(N.gnd, node, n1, Z02, er, wl/4)
    N.transmissionline(N.gnd, node, n2, Z02, er, wl/4)
    N.impedance(n1,n2,2*Z0)
    return [n1, n2]



class Router:
    """
    The Router class is a utility class for quickly designing RF circuits.
    It is designed to work with a Network object and provide a set of
    common RF circuit design patterns.

    Parameters:
    -----------
    N : Network
        The Network object to build the circuit on.
    er : float
        The effective relative permittivity of the substrate.
    Z0 : float
        The characteristic impedance of the transmission lines.
    design_f0 : float
        The design frequency of the circuit.
    VSWR : float
        The voltage standing wave ratio of the circuit components. This will apply random variations each time the circuit is built.
    """
    

    def __init__(self, N: Network, er: float, Z0: float, design_f0: float = 1e9, VSWR: float = 1):

        self.er = er
        self._Z0 = Z0
        self.N: Network = N
        self.f0 = design_f0
        self.VSWR = VSWR

    @property
    def Z0(self) -> float:
        return Z0_VSWR(self._Z0, self.VSWR)
    
    @property
    def _quarter_wavelength(self) -> float:
        return 0.25 * self.f0 / (299792458 * np.sqrt(self.er))
    
    def line(self, node: Node, L: float, Z0: float = None) -> Node:

        if Z0 is None:
            Z0 = self.Z0
        new_node = self.N.node()

        self.N.transmissionline(self.N.gnd, node, new_node, Z0, self.er, L)

        return new_node
    
    def lines(self, nodes: list[Node], L: float, Z0: float = None) -> list[Node]:
        return [self.line(n, L, Z0=Z0) for n in nodes]
    
    def quarter_lambda_section(self, node: Node, Z0: float = None) -> Node:
        return self.line(self.node, self._quarter_wavelength, Z0=Z0)
    
    def unbalanced_splitter(self, node: Node) -> List[Node]:
        return unbalanced_splitter(self.N, node, self.Z0, self.f0, self.er)
    
    def unbalanced_splitters(self, nodes: list[Node]) -> List[Node]:
        return reduce(lambda a,b: a+b, [self.unbalanced_splitter(n) for n in nodes])
    
    def balanced_splitter(self, node: Node) -> List[Node]:
        return balanced_splitter(self.N, node, self.Z0, self.f0, self.er)
    
    def balanced_splitters(self, nodes: list[Node]) -> List[Node]:
        return reduce(lambda a,b: a+b, [self.balanced_splitter(n) for n in nodes])
    