from .rfcircuit import Network, Node
from enum import Enum
import sympy as sym
from sympy.simplify.fu import TR0,TR8
import numpy as np
from loguru import logger

def even(x):
    """ Returns True if x is even, False otherwise. """
    if x % 2 == 0:
        return True
    return False

def odd(x):
    """ Returns True if x is odd, False otherwise. """
    if x % 2 == 1:
        return True
    return False

class FilterType(Enum):
    """Filter type enumeration.
    This enumeration is used to specify the type of filter to create.
    
    constants:
    -----------
    CHEBYCHEV : int
        Chebyshev filter.
    BUTTERWORTH : int
    """
    CHEBYCHEV = 1
    BUTTERWORTH = 2


class BandType(Enum):
    """Band type enumeration.
    This enumeration is used to specify the type of band to create.
    
    constants:
    -----------
    HIGHPASS : int
        Highpass filter.
    LOWPASS : int
        Lowpass filter.
    BANDPASS : int
        Bandpass filter.
    BANDSTOP : int
        Bandstop filter.
    """
    HIGHPASS = 1
    LOWPASS = 2
    BANDPASS = 3
    BANDSTOP = 4


class CauerType(Enum):
    """Cauer type enumeration.

    constants:
    -----------
    TYPE1 : int
        Type 1 Passband ripple
    TYPE2 : int
        Type 2 Stopband ripple

    """
    TYPE1 = 1
    TYPE2 = 2


def prototype_chebychev(order: int, ripple: float) -> tuple[np.ndarray, float]:
    """Generates the Chebyshev prototype filter G-coefficients.

    Parameters
    ----------
    order : int
        The order of the filter.
    ripple : float
        The ripple factor in dB.

    Returns
    -------
    tuple
        A tuple containing the G-coefficients and the load impedance scaler
    
    """
    N = order
    
    Gk = [i for i in range(N + 2)]
    Ak = [i for i in range(N + 2)]
    Bk = [i for i in range(N + 2)]

    B = np.log(1 / np.tanh(ripple * np.log(10) / 40))
    y = np.sinh(B / (2 * N))

    for k in Gk:
        Ak[k] = np.sin((2 * k - 1) * np.pi / (2 * N))
        Bk[k] = y**2 + np.sin(k * np.pi / N) ** 2
        if k == 0:
            Gk[k] = 1
        elif k == 1:
            Gk[k] = 2 * Ak[1] / y
        elif k <= N:
            Gk[k] = 4 * Ak[k - 1] * Ak[k] / (Bk[k - 1] * Gk[k - 1])
        elif k == N + 1:
            if k % 2 == 0:
                Gk[k] = 1
            else:
                Gk[k] = 1 / np.tanh(B / 4) ** 2
    return Gk[1:-1], Gk[-1]


def prototype_butterworth(order: int) -> tuple[np.ndarray, float]:
    """Generates the Butterworth prototype filter G-coefficients.
    
    Parameters
    ----------
    order : int
        The order of the filter.
    
    Returns
    -------
    tuple
        A tuple containing the G-coefficients and the load impedance scaler (for generality, the scaler is always 1)
    
    """

    N = order
    Gk = [i + 1 for i in range(N)]

    for i in Gk:
        Gk[i - 1] = 2 * np.sin(np.pi * (2 * i - 1) / (2 * N))

    return Gk, 1


def gen_cheb_poly(n: int):
    """Generates the nth order Chebyshev polynomial."""
    x = sym.symbols('x')
    if n==0:
        output = 1
    elif n==1:
        T1 = x
        output = T1
    else:
        Tnm1 = x
        Tnm2 = 1
        Tn = 1
        for n in range(n-1):
            Tn = 2*x*Tnm1-Tnm2
            Tnm2 = Tnm1
            Tnm1 = Tn
        output = Tn
    theta, secm = sym.symbols('theta secm')
    output = sym.expand(output,x).subs(x,sym.cos(theta)*secm)
    for i in range(n):
        output = TR0(sym.expand(TR8(output)))
    return sym.collect(output, theta, exact=None)

def impedance_transformer_cheb(Z1: float, Z2: float, A: float, N: int):
    """Generates the impedances for an impedance transformer using Chebyshev polynomials.
    
    Parameters:
    -----------
    Z1 : float
        The characteristic impedance of the first port.
    Z2 : float
        The characteristic impedance of the second port.
    A : float
        The ripple factor in dB.
    N : int
        The number of sections in the transformer.

    """
    theta = sym.symbols('theta')
    func = 2
    fout = 0
    i = 0
    for n in range(N//2+1):
        Gs = sym.symbols(f'Gamma{i}')
        fout = fout + func*Gs*sym.cos((N-n*2)*theta)
        i = i+1
    
    cheb_poly = gen_cheb_poly(N)
    gammas = []
    cheb_poly_output = cheb_poly
    gamma_poly_output = fout
    for i in range(0,N//2+1):
        mult = N-i*2
        symbol = sym.Symbol(f'Gamma{i}')
        T1 = fout.coeff(sym.cos(mult*theta))
        T2 = A*cheb_poly.coeff(sym.cos(mult*theta))
        if mult==0:
            T1 = gamma_poly_output
            T2 = A*cheb_poly_output*2
        gamma = sym.solve(T1-T2, symbol)[0]
        secmval = np.cosh(1/N*np.arccosh((1/(2*A))*np.log(Z2/Z1)))
        gamma_val = gamma.subs(sym.Symbol('secm'), secmval)
        gammas.append(gamma_val)
        cheb_poly_output = cheb_poly_output - cheb_poly.coeff(sym.cos(mult*theta))*sym.cos(mult*theta)
        gamma_poly_output = gamma_poly_output - fout.coeff(sym.cos(mult*theta))*sym.cos(mult*theta)
    gammas = gammas + list(reversed(gammas))[((N+1)%2):]
    impedances = [Z1,]
    Zlast = Z1
    for g in gammas[:-1]:
        Zsec = Zlast*np.exp(2.0*float(g))
        impedances.append(Zsec)
        Zlast = Zsec
    impedances.append(Z2)
    return impedances


class Filtering:

    def __init__(self, N: Network):
        self.N = N 

    def impedance_transformer(self,
                              gnd: Node,
                              port1: Node,
                              port2: Node,
                              Z01: float,
                              Z02: float,
                              fc: float = None,
                              Nsections: int = 2,
                              ripple: float = 0.05,
                              er: float = 1,
                              frange: tuple[float, float] = None
                              ):
        """
        Designs an impedance transformer using Chebyshev polynomials.
        Parameters:
        -----------
        gnd : Node
            The ground node.
        port1 : Node
            The first port node.
        port2 : Node
            The second port node.
        Z01 : float
            The characteristic impedance of the first port.
        Z02 : float
            The characteristic impedance of the second port.
        fc : float, optional
            The center frequency of the transformer. If `frange` is provided, `fc` is computed as the center of `frange`.
        Nsections : int, optional
            The number of sections in the transformer. Default is 2.
        ripple : float, optional
            The ripple factor for the Chebyshev polynomial. Default is 0.05.
        er : float, optional
            The relative permittivity of the medium. Default is 1.
        frange : tuple[float, float], optional
            The frequency range for the transformer. If provided, `fc` and `ripple` are computed based on this range.
        Returns:
        --------
        tuple
            A tuple containing the list of impedances for each section and the computed ripple factor.
        Notes:
        ------
        This function uses Chebyshev polynomials to design an impedance transformer with the specified parameters.
        """
        
        if frange is not None:
            f1, f2 = frange
            fc = 0.5*(f1+f2)
            A = np.cosh(Nsections*np.arccosh(1/np.cos(np.pi/4*(2-(f2-f1)/(fc)))))
            ripple = 1/(2*A) * np.log(Z02/Z01)
            logger.info(f'Computed Ripple = {ripple}')

        l0 = 299792458/fc * 0.25 * (1/np.sqrt(er))
        imps = impedance_transformer_cheb(Z01, Z02, ripple, Nsections)[1:-1]
        nodes = [self.N.named_node('TransformerNode') for i in range(len(imps)-1)]

        nodes = [port1, ] + nodes + [port2,]

        for Z, n1, n2 in zip(imps, nodes[:-1], nodes[1:]):
            logger.debug(f'Transmisison line from {n1} to {n2} with Z0={Z}')
            self.N.transmissionline(gnd, n1, n2, Z, er, l0)
        return imps, ripple
    
    def cauer_filter(
        self,
        gnd: Node,
        port1: Node,
        port2: Node,
        fc: float,
        bw: float,
        order: int = 2,
        ripple: float = 0.05,
        kind: FilterType = FilterType.BUTTERWORTH,
        type: BandType = BandType.LOWPASS,
        Z0: float = 50,
        cauer_type: CauerType = CauerType.TYPE2,
        chebychev_correction: bool = False,
    ):
        """Creates a Cauer ladder filter.
        This function will create a Cauer ladder filter with the specified parameters between port1 and port2.

        Parameters
        ----------
        gnd : Node
            The ground node.
        port1 : Node
            The input port node.
        port2 : Node
            The output port node.
        fc : float
            The cutoff frequency of the filter.
        bw : float
            The bandwidth of the filter.
        order : int, optional
            The order of the filter, by default 2.
        ripple : float, optional
            The ripple of the filter, by default 0.05.
        kind : FilterType, optional
            The type of filter, by default FilterType.BUTTERWORTH.
        type : BandType, optional
            The type of band, by default BandType.LOWPASS.
        Z0 : float, optional
            The reference impedance, by default 50.
        cauer_type : CauerType, optional
            The type of Cauer filter, by default CauerType.TYPE2.
        chebychev_correction : bool, optional
            Whether to apply Chebychev correction, by default False. If turned to true, the cutoff frequency and bandwidth will be corrected to match the ripple.
        
        """
        cauernodes = [port1]
        Nint = 0
        if cauer_type == CauerType.TYPE2:
            Nint = np.floor((order - 1) / 2)
        elif cauer_type == CauerType.TYPE1:
            Nint = np.floor((order - 2) / 2)

        Nint = int(Nint)
        for i in range(0, Nint):
            cauernodes.append(self.N.named_node("V_internal_"))
        cauernodes.append(port2)

        Capacitors = []
        Inductors = []

        w_cutoff = 2 * np.pi * fc
        w0_bandpass = w_cutoff
        delta_omega = 2 * np.pi * bw

        Qfactor = fc / bw

        LoadImpedanceScaler = 1

        if kind == FilterType.BUTTERWORTH:
            Gk, LoadImpedanceScaler = prototype_butterworth(order)
        elif kind == FilterType.CHEBYCHEV:
            Gk, LoadImpedanceScaler = prototype_chebychev(order, ripple)
            if chebychev_correction:
                epsilon = np.sqrt(10 ** (ripple / 10) - 1)

                if type is BandType.HIGHPASS or type is BandType.BANDSTOP:
                    delta_omega = delta_omega * np.cosh(
                        1 / order * np.arccosh(1 / epsilon)
                    )
                    w_cutoff = w_cutoff * np.cosh(1 / order * np.arccosh(1 / epsilon))
                else:
                    delta_omega = delta_omega / np.cosh(
                        1 / order * np.arccosh(1 / epsilon)
                    )
                    w_cutoff = w_cutoff / np.cosh(1 / order * np.arccosh(1 / epsilon))
                Qfactor = w0_bandpass / delta_omega

        add_counter = 0

        if cauer_type == CauerType.TYPE1:
            testf = odd
            LoadImpedanceScaler = 1 / LoadImpedanceScaler
        elif cauer_type == CauerType.TYPE2:
            testf = even
            add_counter = 1

        for i, gk in enumerate(Gk):
            k = i + 1
            node1 = int(np.floor((i + add_counter) / 2))
            node2 = node1 + 1
            if type == BandType.LOWPASS:
                if testf(k):
                    C = self.N.capacitor(gnd, cauernodes[node1], gk / (w_cutoff * Z0))
                    Capacitors.append(C)
                else:
                    L = self.N.inductor(
                        cauernodes[node1], cauernodes[node2], (gk * Z0) / w_cutoff
                    )
                    Inductors.append(L)
            elif type == BandType.HIGHPASS:
                if testf(k):
                    L = self.N.inductor(gnd, cauernodes[node1], Z0 / (gk * w_cutoff))
                    Inductors.append(L)
                else:
                    C = self.N.capacitor(
                        cauernodes[node1], cauernodes[node2], 1 / (w_cutoff * gk * Z0)
                    )
                    Capacitors.append(C)
            elif type == BandType.BANDPASS:
                if testf(k):
                    Cpar = Qfactor * gk / (w0_bandpass * Z0)
                    Lpar = Z0 / (gk * w0_bandpass * Qfactor)
                    Cp = self.N.capacitor(gnd, cauernodes[node1], Cpar)
                    Lp = self.N.inductor(gnd, cauernodes[node1], Lpar)
                    Capacitors.append(Cp)
                    Inductors.append(Lp)
                else:
                    xnode = self.N.named_node("V_series_")
                    Cseries = 1 / (w0_bandpass * gk * Z0 * Qfactor)
                    Lseries = Qfactor * gk * Z0 / w0_bandpass
                    Cser = self.N.capacitor(cauernodes[node1], xnode, Cseries)
                    Lser = self.N.inductor(xnode, cauernodes[node2], Lseries)
                    Capacitors.append(Cser)
                    Inductors.append(Lser)
            elif type is BandType.BANDSTOP:
                if testf(k):
                    xnode = self.N.named_node("V_series_")
                    Cpar = gk / (Qfactor * w0_bandpass * Z0)
                    Lpar = Qfactor * Z0 / (gk * w0_bandpass)
                    Cp = self.N.capacitor(gnd, xnode, Cpar)
                    Lp = self.N.inductor(xnode, cauernodes[node1], Lpar)
                    Capacitors.append(Cp)
                    Inductors.append(Lp)
                else:
                    Cseries = Qfactor / (w0_bandpass * gk * Z0)
                    Lseries = gk * Z0 / (w0_bandpass * Qfactor)
                    Cser = self.N.capacitor(cauernodes[node1], cauernodes[node2], Cseries)
                    Lser = self.N.inductor(cauernodes[node1], cauernodes[node2], Lseries)
                    Capacitors.append(Cser)
                    Inductors.append(Lser)
        return Capacitors, Inductors, LoadImpedanceScaler
    
