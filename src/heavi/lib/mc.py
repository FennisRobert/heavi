
from .libgen import BaseTwoPort
from ..numeric import Random, SimParam, MonteCarlo
from ..transformations import VSWR_to_S11, Z_to_S11
import numpy as np


class TransmissionLine(BaseTwoPort):

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
        

# def random_transmissionline(self, gnd: Node, port1: Node, port2: Node, 
                                
    #                             Z0: float,
    #                             er: float,
    #                             L: float,
    #                             tand: float = 0,
    #                             transmission_loss: float = 0,
    #                             reference_frequency: float = 1e9) -> Component:
        
    #     """
    #     Generates a random transmission line component with specified parameters.
    #     Parameters:
    #     -----------
    #     gnd (Node): The ground node of the transmission line.
    #     port1 (Node): The first port node of the transmission line.
    #     port2 (Node): The second port node of the transmission line.
    #     Z0 (float): The characteristic impedance of the transmission line.
    #     er (float): The relative permittivity of the transmission line.
    #     L (float): The length of the transmission line.
    #     tand (float, optional): The loss tangent of the transmission line. Defaults to 0.
    #     transmission_loss (float, optional): The transmission loss of the line. Defaults to 0.
    #     reference_frequency (float, optional): The reference frequency for the transmission line. Defaults to 1e9 Hz.
    #     Returns:
    #     --------
    #     Component: A transmission line component with the specified parameters.
    #     """
    #     tand = parse_numeric(tand)
    #     Z0v = parse_numeric(Z0)
    #     transmission_loss = parse_numeric(transmission_loss)
    #     reference_frequency = parse_numeric(reference_frequency)
    #     Lv = parse_numeric(L)
    #     er = parse_numeric(er)
        
    #     tandsimval = Function(lambda f: tand(f) + (transmission_loss(f)/Lv(f)) * 299792458/(reference_frequency(f)*27.28754*np.sqrt(er(f))))
        
    #     ersimval = Function(lambda f: er(f) * (1 - tandsimval(f)*1j))

    #     return self.transmissionline(gnd, port1, port2, Z0v, ersimval, Lv)


    # def random_two_port(
        
    #     self, gnd: Node, port1: Node, port2: Node, VSWR: float | SimValue, Loss: float | SimValue, Z0: float | SimValue
    # ) -> Component:
    #     """
    #     Generates a random two-port network component with specified parameters.
    #     Parameters:
    #     -----------
    #     gnd : Node
    #         The ground node of the circuit.
    #     port1 : Node
    #         The first port node of the two-port network.
    #     port2 : Node
    #         The second port node of the two-port network.
    #     VSWR : float
    #         Voltage Standing Wave Ratio, used to determine the reflection coefficient.
    #     Loss : float
    #         Insertion loss in dB, used to determine the magnitude of S21.
    #     Z0 : float
    #         Characteristic impedance of the network.
    #     Returns:
    #     --------
    #     Component
    #         A two-port network component with the specified parameters.
    #     """
    #     lossmag = 10 ** (-Loss / 20)
    #     vswr1 = randmag(1, VSWR)
    #     vswr2 = randmag(1, VSWR)

    #     reflect1 = (vswr1 - 1) / (vswr1 + 1)
    #     reflect2 = (vswr2 - 1) / (vswr2 + 1)

    #     A, B = randphase(), randphase()

    #     def S11(f):
    #         return A * reflect1
    #     def S22(f):
    #         return B * reflect2
    #     def S21(f):
    #         return B * lossmag
    #     return self.two_port_reciprocal(gnd, port1, port2, S11, S21, S22, Z0)


        # def random_power_splitter(
    #     self,
    #     gnd: Node,
    #     pin: Node,
    #     pouts: list[Node],
    #     port_VSWR: float = 1,
    #     transmission_loss: float = 0,
    #     Z0: float = 50.0,
    #     isolation: float = -80,
    #     transmission_phase: float = 0
    # ) -> Component:
    #     """
    #     Creates an (M+1)-port 'quasi-ideal' stochastic power splitter:
    #     - One input port (pin)
    #     - M = len(pouts) output ports
    #     Each port gets a random reflection based on a uniform VSWR in [1, port_VSWR].
    #     The total transmission from pin -> each pout is split equally in power (1/M),
    #     then scaled by 'transmission_loss' (dB). Random phases are added to reflections and transmissions.

    #     Parameters
    #     ----------
    #     gnd : Node
    #         Common ground node.
    #     pin : Node
    #         The input port node.
    #     pouts : list[Node]
    #         The list of output port nodes.
    #     port_VSWR : float, optional
    #         The maximum possible VSWR for each port, by default 1 (i.e., no mismatch).
    #     transmission_loss : float, optional
    #         Transmission loss in dB (applied equally to all outputs), by default 0.
    #     Z0 : float, optional
    #         Reference impedance for the S-parameters, by default 50 Ohms.

    #     Returns
    #     -------
    #     Component
    #         The N-port component that represents this power splitter.
    #     """
    #     Z0 = parse_numeric(Z0)
    #     transmission_phase = parse_numeric(transmission_phase)
    #     isolation = parse_numeric(Z0)

    #     # Number of output ports
    #     M = len(pouts)
    #     # Total ports = 1 input + M outputs
    #     N = M + 1

    #     # Gather all ports in a list: port 0 = pin, port i>0 => pouts[i-1]
    #     all_ports = [pin] + pouts

    #     # Convert the dB loss into a linear magnitude factor
    #     loss_mag = 10 ** (-transmission_loss / 20)
    #     isolation_amp = 10**(-np.abs(isolation) / 20)

    #     # Build an NxN array of callables for S-parameters
    #     # Sparam[i][j] is a function f -> complex
    #     Sparam = [[None for _ in range(N)] for _ in range(N)]

    #     # 1) Random reflection for each port (VSWR-based, random phase)
    #     for i in range(N):
    #         vswr_rand = randmag(1, port_VSWR)     # uniform random in [1, port_VSWR]
    #         reflect_mag = (vswr_rand - 1) / (vswr_rand + 1)
    #         phase = randphase()
    #         Sparam[i][i] = (  # Siila
    #             lambda mag=reflect_mag, ph=phase:
    #                 (lambda f: mag * ph * np.ones_like(f))  # frequency-independent
    #         )()

    #     # 2) Set up transmissions from port 0 -> each output, and reciprocal
    #     #    The amplitude for each output is loss_mag / sqrt(M), plus random phase
    #     if M > 0:
    #         amp_each = loss_mag / np.sqrt(M)
    #         for i in range(1, N):
    #             # random phase for this output
    #             phase_ij = 1
    #             val_ij = amp_each * phase_ij * np.sqrt(1-np.abs(Sparam[i][i](1))**2) # complex amplitude

    #             # S[i,0] and S[0,i] must match for reciprocity
    #             Sparam[i][0] = (lambda v=val_ij: (lambda f: v * np.ones_like(f)))()
    #             Sparam[0][i] = (lambda v=val_ij: (lambda f: v * np.ones_like(f)))()

    #     # 3) Zero out cross-coupling between outputs (ideal isolation):
    #     #    S[i,j] = 0 for i != j, excluding anything with 0 we already set
    #     for i in range(1, N):
    #         for j in range(1, N):
    #             if i != j:
    #                 phase = randphase()
    #                 Sparam[i][j] = (lambda f: phase*isolation_amp* np.ones_like(f))

    #     # Now we have an NxN array of S-parameter callables
    #     # Pass them into the N-port reciprocal function
    #     # This will create one N-port Component with Y-parameters
    #     comp = self.n_port_S(
    #         gnd=gnd,
    #         nodes=all_ports,    # port0=pin, port1..M = pouts
    #         Sparam=Sparam,
    #         Z0=Z0
    #     )
    #     return comp