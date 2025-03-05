from .libgen import BaseComponent, BaseTwoPort
from ..numeric import SimParam, enforce_simparam, Function
from ..rfcircuit import Component
import numpy as np

class Inductor(BaseTwoPort):

    def __init__(self, inductance: float, parasitic_capacitance: float = 0, parasitic_resistance: float = 0):
        super().__init__()
        self.inductance: float = inductance
        self.parasitic_capacitance: float = parasitic_capacitance
        self.parasitic_resistance: float = parasitic_resistance
        self.component: Component = None

    def __on_connect__(self):
        pL = enforce_simparam(self.inductance)
        pC = enforce_simparam(self.parasitic_capacitance)
        pR = enforce_simparam(self.parasitic_resistance)

        if pC == 0 and pR == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(1j * 2 * np.pi * f * pL(f))
        elif pC == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(1j * 2 * np.pi * f * pL(f) + pR(f))
        elif pR == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(1j * 2 * np.pi * f * pL(f)) + 1j * 2 * np.pi * f * pC(f)
        else:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(1j * 2 * np.pi * f * pL(f) + pR(f)) + 1j * 2 * np.pi * f * pC(f)
        self.component = self.network.admittance(self.node(1), self.node(2), Function(Y))\
                                                    .set_metadata(name='Lumped Inductor',
                                                                  unit='H',
                                                                  value=self.inductance,
                                                                  inductance=self.inductance, 
                                                                  parasitic_capacitance=self.parasitic_capacitance, 
                                                                  parasitic_resistance=self.parasitic_resistance)

    
class Capacitor(BaseTwoPort):

    def __init__(self, capacitance: float, parasitic_inductance: float = 0, parasitic_resistance: float = 0):
        super().__init__()
        self.capacitance: float = capacitance
        self.parasitic_inductance: float = parasitic_inductance
        self.parasitic_resistance: float = parasitic_resistance
        self.component: Component = None

    def __on_connect__(self):
        pC = enforce_simparam(self.capacitance)
        pL = enforce_simparam(self.parasitic_inductance)
        pR = enforce_simparam(self.parasitic_resistance)

        if pL == 0 and pR == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1j * 2 * np.pi * f * pC(f)
        elif pL == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(1/(1j * 2 * np.pi * f * pC(f)) + pR(f))
        elif pR == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(1/(1j * 2 * np.pi * f * pC(f)) + 1j * 2 * np.pi * f * pL(f))
        else:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(1/(1j * 2 * np.pi * f * pC(f)) + pR(f) + 1j * 2 * np.pi * f * pL(f))
        
        self.component = self.network.admittance(self.node(1), self.node(2), Function(Y))\
                                                    .set_metadata(name='Lumped Capacitor',
                                                                  unit='F',
                                                                  value=self.capacitance,
                                                                  capacitance=self.capacitance, parasitic_inductance=self.parasitic_inductance, parasitic_resistance=self.parasitic_resistance)

class Resistor(BaseTwoPort):

    def __init__(self, resistance: float, parasitic_capacitance: float = 0, parasitic_inductance: float = 0):
        super().__init__()
        self.resistance: float = resistance
        self.component: Component = None
        self.parasitic_capacitance: float = parasitic_capacitance
        self.parasitic_inductance: float = parasitic_inductance


    def __on_connect__(self):
        pR = enforce_simparam(self.resistance)
        pC = enforce_simparam(self.parasitic_capacitance)
        pL = enforce_simparam(self.parasitic_inductance)

        if pC == 0 and pL == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/pR(f)
        elif pC == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(pR(f) + 1j * 2 * np.pi * f * pL(f))
        elif pL == 0:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(pR(f)) + 1j * 2 * np.pi * f * pC(f)
        else:
            def Y(f: np.ndarray) -> np.ndarray:
                return 1/(pR(f) + 1j * 2 * np.pi * f * pL(f)) + 1j * 2 * np.pi * f * pC(f)

        self.component = self.network.admittance(self.node(1), self.node(2), Function(Y)) \
                                                    .set_metadata(name='Lumped Resistor',
                                                                  unit='Ω',
                                                                  value=self.resistance,
                                                                  resistance=self.resistance, 
                                                                  parasitic_capacitance=self.parasitic_capacitance, 
                                                                  parasitic_inductance=self.parasitic_inductance)

class Impedance(BaseTwoPort):

    def __init__(self, impedance: float | SimParam):
        super().__init__()
        self.impedance: SimParam = enforce_simparam(impedance)
        self.component: Component = None

    def __on_connect__(self):
        self.component = self.network.impedance(self.node(1), self.node(2), self.impedance.inverse)\
                                                    .set_metadata(name='Lumped Impedance',
                                                                  unit='Ω',
                                                                  value=self.impedance,
                                                                  impedance=self.impedance)

class Admittance(BaseTwoPort):
    
    def __init__(self, admittance: float | SimParam):
        super().__init__()
        self.admittance: SimParam = enforce_simparam(admittance)
        self.component: Component = None

    def __on_connect__(self):
        self.component = self.network.admittance(self.node(1), self.node(2), self.admittance)\
                                                    .set_metadata(name='Lumped Admittance',
                                                                  unit='Siemens',
                                                                  value=self.admittance,
                                                                  admittance=self.admittance)

class TransmissionLineGrounded(BaseTwoPort):
    def __init__(self, Z0: float, length: float, er: float = 1):
        super().__init__()
        self.Z0: float = Z0
        self.er: float = er
        self.length: float = length
        self.component: Component = None

    def __on_connect__(self):
        pZ0 = enforce_simparam(self.Z0)
        per = enforce_simparam(self.er)
        plength = enforce_simparam(self.length)
        comp = self.network.TL(self.node(1), self.node(2), lambda f: 2*np.pi*f/299792458 * np.sqrt(per(f)), plength, pZ0)
        comp.set_metadata(name='Transmission Line',
                          unit='Ω',
                          Z0=self.Z0,
                          er=self.er,
                          length=self.length)

class TransmissionLine(BaseTwoPort):

    def __init__(self, Z0: float, length: float, er: float = 1):
        super().__init__()
        self.Z0: float = Z0
        self.er: float = er
        self.length: float = length
        self.component: Component = None

    def __on_connect__(self):
        pZ0 = enforce_simparam(self.Z0)
        per = enforce_simparam(self.er)
        plength = enforce_simparam(self.length)
        self.network.TL(self.node(1), self.node(2), lambda f: 2*np.pi*f/299792458 * np.sqrt(per(f)), plength, pZ0)

L = Inductor
C = Capacitor
Z = Impedance
Y = Admittance
TL = TransmissionLine
TLg = TransmissionLineGrounded
# class StandardLinear(BaseRouter):

#     def L(self, inductance: float, parasitic_capacitance: float = 0, parasitic_resistance: float = 0) -> Inductor:
#         return Inductor(inductance, parasitic_capacitance, parasitic_resistance).attach(self.node)
    
#     def C(self, capacitance: float, parasitic_inductance: float = 0, parasitic_resistance: float = 0) -> Capacitor:
#         return Capacitor(capacitance, parasitic_inductance, parasitic_resistance).attach(self.node)
    
#     def R(self, resistance: float, parasitic_capacitance: float = 0, parasitic_inductance: float = 0) -> Resistor:
#         return Resistor(resistance, parasitic_capacitance, parasitic_inductance).attach(self.node)
    
#     def Z(self, impedance: float | SimParam) -> Impedance:
#         return Impedance(impedance).attach(self.node)
    
#     def Y(self, admittance: float | SimParam) -> Admittance:
#         return Admittance(admittance).attach(self.node)
    
#     def TL(self, Z0: float, length: float, er: float = 1) -> TransmissionLine:
#         return TransmissionLine(Z0, length, er).attach(self.node)


# class Lumped(BaseNetwork):

#     def port(self, impedance: float | SimParam = 50) -> StandardLinear:
#         node = self.network.port(impedance)
#         return BaseRouter(node, self.network)