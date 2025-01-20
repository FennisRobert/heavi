import numpy as np
from .rfcircuit import Network, Node, ComponentType
from typing import Union, Callable, Optional

c0 = 299792458

class Library:
    """ 
    The Library class extends the standard Network class with a set of
    predefined or higher-order definitions of common components (R, L, C, etc.).
    
    The methods in this class leverage `network.impedance(...)` to build up
    custom frequency-dependent or compound component models.
    """

    def __init__(self, N: Network):
        self.N = N

    # ------------------------------------------------------------------------
    # 1. Basic, Ideal Components
    # ------------------------------------------------------------------------
    def resistor(self, node1: Node, node2: Node, R: float):
        """
        Creates an ideal resistor between two nodes.
        
        Parameters:
        -----------
        node1, node2 : Node
            The two nodes to connect the resistor between.
        R : float
            The resistance value in ohms (Ω).
        """
        def Z_of_f(_: float) -> complex:
            # Frequency-independent real resistor
            return R
        
        return self.N.impedance(node1, node2, Z_of_f)

    def capacitor(self, node1: Node, node2: Node, C: float):
        """
        Creates an ideal capacitor between two nodes.
        
        Parameters:
        -----------
        node1, node2 : Node
            The two nodes to connect the capacitor between.
        C : float
            The capacitance in farads (F).
        """
        def Z_of_f(f: float) -> complex:
            # Zc = 1 / (j*2πf*C). For f=0, this is infinite impedance.
            # We can handle f=0 by returning a large number or let it naturally blow up.
            w = 2.0 * np.pi * f
            return 1.0 / (1j * w * C)
        
        return self.N.impedance(node1, node2, Z_of_f)

    def inductor(self, node1: Node, node2: Node, L: float, ESR: float = 0.0):
        """
        Creates an inductor with optional ESR (Equivalent Series Resistance).
        
        Parameters:
        -----------
        node1, node2 : Node
            The two nodes to connect the inductor between.
        L : float
            The inductance in henries (H).
        ESR : float, optional
            A small series resistance to model real inductor losses.
        """
        def Z_of_f(f: float) -> complex:
            w = 2.0 * np.pi * f
            # Z = R + j w L
            return ESR + 1j * w * L
        
        return self.N.impedance(node1, node2, Z_of_f)

    # ------------------------------------------------------------------------
    # 2. Compound R with Parasitics
    # ------------------------------------------------------------------------
    def R(self, node1: Node, node2: Node, R: float, Cpar: float = 0.0, Lpar: float = 0.0):
        """
        Creates a resistor between two nodes with optional parasitic
        parallel capacitance (Cpar) and series inductance (Lpar).
        
        In many real-world SMD resistors, there can be a parasitic inductance
        (particularly at high frequencies), and some stray/parallel capacitance.
        
        Parameters:
        -----------
        node1, node2 : Node
            The two nodes to connect the R-C-L.
        R : float
            The resistor value in ohms (Ω).
        Cpar : float, optional
            Parasitic (often small) capacitance in farads (F).
        Lpar : float, optional
            Parasitic (often very small) inductance in henries (H).
        """
        # Create the nominal resistor
        self.resistor(node1, node2, R)

        # Optionally add the parasitic capacitor in parallel
        if Cpar:
            self.capacitor(node1, node2, Cpar)

        # Optionally add the parasitic inductor in series
        # In reality, "series" inductance is usually modeled as part of the same lead path,
        # but for simplicity we'll just add it in the same two nodes.
        if Lpar:
            self.inductor(node1, node2, Lpar)

    # ------------------------------------------------------------------------
    # 3. More Advanced Models
    # ------------------------------------------------------------------------
    def inductor_srf(self, node1: Node, node2: Node, L: float,
                     f_srf: float,
                     ESR: float = 0.0):
        """
        Creates an inductor that includes a parallel capacitance to
        approximate self-resonance at frequency f_srf, plus an ESR.
        
        The typical assumption is that the self-resonant frequency (f_srf)
        arises from the main inductance L and some parasitic *parallel* capacitance Cpar.
        
        We compute Cpar via:  f_srf = 1 / (2π * sqrt(L*Cpar))  =>  Cpar = 1 / ((2π*f_srf)^2 * L)
        
        The final model is:
            Z = (ESR + jωL) || (1 / jωCpar)
        i.e., the inductor with series ESR is in parallel with the parasitic capacitor.
        
        Parameters:
        -----------
        node1, node2 : Node
            The two nodes to connect the inductor between.
        L : float
            The inductance in henries (H).
        f_srf : float
            The self resonance frequency in Hz.
        ESR : float, optional
            Equivalent series resistance of the inductor.
        """
        Cpar = 1.0 / ((2.0 * np.pi * f_srf)**2 * L)

        def Z_of_f(f: float) -> complex:
            w = 2.0 * np.pi * f
            # Series branch: ESR + j w L
            Z_series = ESR + 1j * w * L
            # Parallel capacitor branch: 1 / (j w Cpar)
            if f == 0:
                # At f=0, the capacitor is open (Z->∞), so the inductor branch is just ESR
                return Z_series  
            Z_cap = 1.0 / (1j * w * Cpar)

            # Combine in parallel: Z = 1 / (1/Z_series + 1/Z_cap)
            return 1.0 / (1.0 / Z_series + 1.0 / Z_cap)

        return self.N.impedance(node1, node2, Z_of_f)

    def capacitor_esr_esl(self, node1: Node, node2: Node, C: float,
                          ESR: float = 0.0, ESL: float = 0.0):
        """
        A capacitor with equivalent series resistance (ESR) and equivalent
        series inductance (ESL). A common real-world capacitor model is:
        
            Z = ESR + jωESL + 1 / (jωC)
        
        All in series. For higher-end modeling, you might add parallel leakage, etc.
        
        Parameters:
        -----------
        node1, node2 : Node
            The two nodes to connect the capacitor between.
        C : float
            The capacitance in farads (F).
        ESR : float, optional
            Equivalent series resistance in ohms (Ω).
        ESL : float, optional
            Equivalent series inductance in henries (H).
        """
        def Z_of_f(f: float) -> complex:
            w = 2.0 * np.pi * f
            # Series combination: ESR + j w ESL + 1/(j w C)
            # Handle the f=0 or w=0 case carefully:
            if f == 0:
                # 1/(j*0*C) -> infinite => open circuit for the capacitive part,
                # so the total is ESR + j*0 => ESR
                return ESR
            Z_c = 1.0 / (1j * w * C)
            return ESR + 1j * w * ESL + Z_c

        return self.N.impedance(node1, node2, Z_of_f)

    # ------------------------------------------------------------------------
    # 4. Example of a "callable-based" generic method
    # ------------------------------------------------------------------------
    def custom_impedance(self,
                         node1: Node,
                         node2: Node,
                         Zfunc: Callable[[float], complex]):
        """
        Attach a custom, user-defined frequency-dependent impedance between
        two nodes. This is useful if you want to do something more exotic.
        
        Parameters:
        -----------
        node1, node2 : Node
            The two nodes to connect.
        Zfunc : Callable
            A function that takes frequency `f` (Hz) and returns a complex
            impedance `Z(f)`.
        
        Example:
            def Z_of_f(f):
                # Just a made-up frequency dependency
                return 1.0 + 1j*(2*np.pi*f*0.1)
            my_lib.custom_impedance(nodeA, nodeB, Z_of_f)
        """
        return self.N.impedance(node1, node2, Zfunc)
