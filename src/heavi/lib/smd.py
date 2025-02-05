from .libgen import BaseComponent, BaseTwoPort
from enum import Enum


class SMDResistorSize(Enum):
    R0402 = "0402"
    R0603 = "0603"
    R0805 = "0805"
    R1206 = "1206"
    # Add other sizes as needed

class SMDCapacitorSize(Enum):
    C0402 = "0402"
    C0603 = "0603"
    C0805 = "0805"
    C1206 = "1206"
    # Add other sizes as needed

class SMDInductorSize(Enum):
    L0402 = "0402"
    L0603 = "0603"
    L0805 = "0805"
    L1206 = "1206"
    # Add other sizes as needed

class SMDResistor(BaseTwoPort):
    """
    A simple SMD resistor model that accounts for parasitic inductance and capacitance 
    based on package size.

    Attributes
    ----------
    resistance : float
        The resistance value in ohms.
    inductance : float
        The parasitic inductance in henries.
    capacitance : float
        The parasitic capacitance in farads.
    """

    def __init__(self, resistance: float, package: SMDResistorSize):
        """
        Parameters
        ----------
        resistance : float
            The nominal resistance value (in ohms).
        package : SMDResistorSize
            The package size of the SMD resistor (e.g., 0402, 0603, etc.).
        """
        super().__init__()
        self.n_nodes = 2
        self.resistance = resistance
        self.package = package
        # Typical empirical parasitic values (approximate):
        # These can vary depending on manufacturer, pad layout, etc.
        # You can refine or expand as necessary.
        package_parasitics = {
            SMDResistorSize.R0402: (0.8e-9, 0.2e-12),  # (L, C)
            SMDResistorSize.R0603: (1.0e-9, 0.3e-12),
            SMDResistorSize.R0805: (1.5e-9, 0.45e-12),
            SMDResistorSize.R1206: (1.8e-9, 0.6e-12),
        }
        
        # Fallback for unknown sizes (or define a default)
        default_inductance = 1.0e-9
        default_capacitance = 0.3e-12
        
        if package in package_parasitics:
            self.inductance, self.capacitance = package_parasitics[package]
        else:
            self.inductance = default_inductance
            self.capacitance = default_capacitance

        self.function = None

    def __on_connect__(self):
        """
        Connects the SMD resistor with its parasitic R, L, and C
        into the network. This method is automatically called
        when the component is attached to the network.
        """
        
        def z_parasitic(f):
            # Z_RLC = R + j*2*pi*f*L + 1/(j*2*pi*f*C)
            w = 2 * 3.141592653589793 * f
            Zr = self.resistance
            Zl = 1j * w * self.inductance
            Zc = 1 / (1j * w * self.capacitance)
            # return Zr + Zl parallel to Zc
            return (Zr+Zl) * Zc / (Zr + Zl + Zc)
        self.function = z_parasitic
        self.network.impedance(self.node(1), self.node(2), z_parasitic, display_value=self.resistance)\
        .set_metadata(name=f'{self.package}SMD Resistor',
                      unit='Ω',
                      value=self.resistance,
                      inductance=self.inductance,
                      capacitance=self.capacitance)

class SMDInductor(BaseTwoPort):
    """
    An SMD inductor with a simple R-L // C parasitic model.
    
    - The inductor has a series ESR (Equivalent Series Resistance) with the inductance.
    - A small parallel capacitance accounts for high-frequency behavior.
    
    Attributes
    ----------
    inductance : float
        The nominal inductance value in henries (H).
    esr : float
        The equivalent series resistance in ohms (Ω).
    parallel_capacitance : float
        The parasitic parallel capacitance in farads (F).
    """

    def __init__(self, inductance: float, package: SMDInductorSize):
        """
        Parameters
        ----------
        inductance : float
            The inductance value in henries.
        package : SMDInductorSize
            The package size of the SMD inductor (e.g. 0402, 0603, etc.).
        """
        super().__init__()
        self.package = package
        self.n_nodes = 2
        self.inductance = inductance
        
        # Typical parasitic values (approximate)
        # Key: SMDInductorSize, Value: (ESR, parallel_capacitance)
        # These are purely illustrative; real values depend on manufacturer & frequency range.
        package_parasitics = {
            SMDInductorSize.L0402: (0.15, 0.3e-12),
            SMDInductorSize.L0603: (0.10, 0.5e-12),
            SMDInductorSize.L0805: (0.08, 0.8e-12),
            SMDInductorSize.L1206: (0.06, 1.2e-12),
        }
        
        # Fallback/default values if package not in dict
        default_esr = 0.1
        default_cpar = 0.5e-12
        
        if package in package_parasitics:
            self.esr, self.parallel_capacitance = package_parasitics[package]
        else:
            self.esr = default_esr
            self.parallel_capacitance = default_cpar

        self.function = None

    def __on_connect__(self):
        """
        Connects the SMD inductor in the network as a single
        frequency-dependent impedance that models:
        
        Z_total = (ESR + jωL) ∥ (1 / jωCpar)
        
        where:
        - ESR is the inductor's equivalent series resistance,
        - L is the nominal inductance,
        - Cpar is the parasitic parallel capacitance.
        """
        import cmath
        import math

        def z_parasitic(f):
            w = 2 * math.pi * f
            
            # Series branch = ESR + jωL
            z_series = self.esr + 1j * w * self.inductance
            
            # Parallel capacitor = 1 / (jωC)
            z_parallel_cap = 1 / (1j * w * self.parallel_capacitance)
            
            # Combine them in parallel:
            # Z_total = (z_series * z_parallel_cap) / (z_series + z_parallel_cap)
            return 1 / (1/(z_series) + 1/(z_parallel_cap))#(z_series * z_parallel_cap) / (z_series + z_parallel_cap)
        self.function = z_parasitic
        # Create the impedance component in the network
        self.network.impedance(self.node(1), self.node(2), z_parasitic, display_value=self.inductance)\
        .set_metadata(name=f'{self.package}SMD Inductor',
                      unit='H',
                      value=self.inductance,
                      esr=self.esr,
                      parallel_capacitance=self.parallel_capacitance)

class SMDCapacitor(BaseTwoPort):
    """
    An SMD capacitor with an ESL-ESR-C series model.
    
    Attributes
    ----------
    capacitance : float
        The nominal capacitance value in farads (F).
    esr : float
        The equivalent series resistance in ohms (Ω).
    esl : float
        The equivalent series inductance in henries (H).
    """

    def __init__(self, capacitance: float, package: SMDCapacitorSize):
        """
        Parameters
        ----------
        capacitance : float
            The capacitance value in farads.
        package : SMDCapacitorSize
            The package size of the SMD capacitor (e.g., 0402, 0603, etc.).
        """
        super().__init__()
        self.package = package
        self.n_nodes = 2
        self.capacitance = capacitance
        
        # Typical parasitic values (approximate)
        # Key: SMDCapacitorSize, Value: (ESR, ESL)
        package_parasitics = {
            SMDCapacitorSize.C0402: (0.1, 0.3e-9),
            SMDCapacitorSize.C0603: (0.08, 0.5e-9),
            SMDCapacitorSize.C0805: (0.07, 0.7e-9),
            SMDCapacitorSize.C1206: (0.05, 1.0e-9),
        }
        
        # Fallback/default values if package not in dict
        default_esr = 0.1
        default_esl = 0.5e-9
        
        if package in package_parasitics:
            self.esr, self.esl = package_parasitics[package]
        else:
            self.esr = default_esr
            self.esl = default_esl

        self.function = None

    def __on_connect__(self):
        """
        Connects the SMD capacitor in the network as a single
        frequency-dependent impedance that models:
        
        Z_total = ESR + jωESL + 1 / (jωC)
        
        (all components in series).
        """
        import cmath
        import math
        
        def z_parasitic(f):
            w = 2 * math.pi * f
            
            # Series components: ESR + jωESL
            z_series = self.esr + 1j * w * self.esl
            
            # Capacitive reactance = 1 / (jωC)
            z_c = 1 / (1j * w * self.capacitance)
            
            # Total series: Z_total = z_series + z_c
            return z_series + z_c
        self.function = z_parasitic
        # Create the impedance component in the network
        self.network.impedance(self.node(1), self.node(2), z_parasitic, display_value=self.capacitance)\
        .set_metadata(name=f'{self.package}SMD Capacitor',
                      unit='F',
                      value=self.capacitance,
                      esr=self.esr,
                      esl=self.esl)