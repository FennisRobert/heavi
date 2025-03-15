
from .libgen import TwoNodeSubCircuit
from ..numeric import Random, SimParam, MonteCarlo
from ..transformations import VSWR_to_S11, Z_to_S11
import numpy as np


class RandomTwoPort(TwoNodeSubCircuit):
    """Generates a random two-port network component with specified parameters.
    Only to be used in S-parameter networks.

    """    
    def __init__(self, 
                 mcsim: MonteCarlo,
                 length: float, 
                 Z0: float,
                 VSWR: float = 1,
                 er: float = 1,
                 attenuation: float = 0,
                 phase_stability: float = 0,
                 amplitude_stability: float = 0,
                 Z0_error: float = 0,
                 nsigma: float = 3,
                 ):
        """ Generates a random two-port network component with specified parameters.

        This class is only intended to be used in S-parameter simulations with Monte Carlo analysis.

        Args:
            mcsim (MonteCarlo): Monte Carlo simulation object.
            length (float): The Electrical length of the two-port network.
            Z0 (float): The characteristic impedance of the two-port network.
            VSWR (float, optional): The limit of the inputs. Defaults to 1.
            er (float, optional): The dielectric constant associated with the electrical length. Defaults to 1.
            attenuation (float, optional): The port-to-port attentuation. Defaults to 0.
            phase_stability (float, optional): The std. of the phase variation per sim. Defaults to 0.
            amplitude_stability (float, optional): The std. of the amplitude variation. Defaults to 0.
            Z0_error (float, optional): The std. of the port input impedance. Defaults to 0.
            nsigma (float, optional): The number of std. caputed with each provided std limit. Defaults to 3.

        """        
        super().__init__()
        self.mc: MonteCarlo = mcsim

        def kz(f):
            return 2*np.pi*f/(299792458)*np.sqrt(er)
        
        A = mcsim.uniform(1, VSWR)
        B = mcsim.uniform(0, 2*np.pi)
        C = mcsim.gaussian(Z0, Z0_error/nsigma)
        D = mcsim.gaussian(0,amplitude_stability/nsigma)
        E = mcsim.gaussian(0,phase_stability/nsigma)
        self.fS11 = lambda f: np.exp(1j*B(f))*VSWR_to_S11(A(f))*Z_to_S11(C(f), Z0)
        self.fS21 = lambda f: (((1-self.fS11(f)**2)**(0.5))*10**(-(attenuation*length)/20) 
                               * (10**(D(f)/20)) 
                               * np.exp(1j*(kz(f)*length+E(f)*np.pi/180)))
        self.fS12 = self.fS21
        self.fS22 = self.fS11
        self.Z0 = Z0
        
    def __on_connect__(self):
        self.network.n_port_S(self.gnd, [self.node(1), self.node(2)], 
                              [[self.fS11, self.fS12], [self.fS21, self.fS22]], 
                              self.Z0)
        
