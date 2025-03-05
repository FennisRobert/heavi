from .libgen import BaseComponent, BaseTwoPort, BaseFourPort, BaseThreePort
from ..numeric import SimParam, enforce_simparam
from ..rfcircuit import Node
import numpy as np
from ..filtering import impedance_transformer_cheb

class Wilkinson(BaseThreePort):

    def __init__(self, Z0: float | SimParam, fc: float | SimParam, er: float | SimParam = 1):
        super().__init__()
        self.Z0 = Z0
        self.fc = fc
        self.er = er
    
    def __on_connect__(self):
        wavelength = 299792458 / (self.fc*np.sqrt(self.er))
        def beta(f):
            return 2*np.pi*f*np.sqrt(self.er)/(299792458)
        L = wavelength / 4

        Z0 = self.Z0 * np.sqrt(2)

        self.network.TL(self.node(1), self.node(2), beta, L, Z0)
        self.network.TL(self.node(1), self.node(3), beta, L, Z0)
        self.network.resistor(self.node(2), self.node(3), Z0*2)

class BranchlineCoupler(BaseFourPort):

    def __init__(self, Z0: float | SimParam, fc: float | SimParam, er: float | SimParam = 1):
        super().__init__()
        self.Z0 = Z0
        self.fc = fc
        self.er = er
    
    def __on_connect__(self):
        wavelength = 299792458 / (self.fc*np.sqrt(self.er))
        L = wavelength / 4
        def beta(f):
            return 2*np.pi*f*np.sqrt(self.er)/(299792458)
        Z0 = self.Z0/np.sqrt(2)

        self.network.TL(self.node(1), self.node(2), beta, L, Z0)
        self.network.TL(self.node(4), self.node(3), beta, L, Z0)
        self.network.TL(self.node(1), self.node(4), beta, L, self.Z0)
        self.network.TL(self.node(3), self.node(3), beta, L, self.Z0)

class RatraceCoupler(BaseFourPort):

    def __init__(self, Z0: float | SimParam, fc: float | SimParam, er: float | SimParam = 1):
        super().__init__()
        self.Z0 = Z0
        self.fc = fc
        self.er = er
    
    def __on_connect__(self):
        wavelength = 299792458 / (self.fc*np.sqrt(self.er))
        L = wavelength / 4

        def beta(f):
            return 2*np.pi*f*np.sqrt(self.er)/(299792458)
        
        Z0 = self.Z0*np.sqrt(2)

        self.network.TL(self.node(1), self.node(2), beta, L, Z0)
        self.network.TL(self.node(2), self.node(3), beta, L, Z0)
        self.network.TL(self.node(3), self.node(4), beta, L, Z0)
        self.network.TL(self.node(1), self.node(4), beta, 3*L, Z0)

    @property
    def sum(self) -> Node:
        return self.node(2)
    
    @property
    def diff(self) -> Node:
        return self.node(4)
    
    @property
    def delta(self) -> Node:
        return self.node(4)
    
    @property
    def p1(self) -> Node:
        return self.node(1)
    
    @property
    def p2(self) -> Node:
        return self.node(3)
    
class Transformer(BaseTwoPort):

    def __init__(self, 
                 Z1: float | SimParam,
                 Z2: float | SimParam,
                 fc: float | SimParam, 
                 N_sections: int = 1,
                 ripple: float | SimParam = 0.05,
                 er: float | SimParam = 1):
        super().__init__()
        self.Z1 = Z1
        self.Z2 = Z2
        self.fc = fc
        self.N_sections = N_sections
        self.ripple = ripple
        self.er = er

    def __on_connect__(self):
        impedances = impedance_transformer_cheb(self.Z1, self.Z2, self.ripple, self.N_sections)[1:-1]
        wavelength = 299792458 / (self.fc * np.sqrt(self.er))
        L = wavelength / 4
        nodes = [self.node(1),] + [self.network.node() for _ in range(self.N_sections-1)] + [self.node(2),]
        for n1, n2, z0 in zip(nodes[:-1],nodes[1:],impedances):
            self.network.TL(n1, n2, lambda f: 2*np.pi*f/299792458 * np.sqrt(self.er), L, float(z0))
        