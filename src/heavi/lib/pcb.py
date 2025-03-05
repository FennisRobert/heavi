
from __future__ import annotations
import numpy as np
from enum import Enum
from .libgen import BaseComponent
from ..rfcircuit import Network, Node
from scipy.optimize import root_scalar
from loguru import logger

u0 = 4*np.pi*1e-7
def coth(x):
    return -np.tan(x + np.pi/2)

def _microstrip_ereff(w: float, h: float, er: float, t: float):
    u = w/h
    T = t/h
    du1 = T/np.pi * np.log(1 + 4*np.exp(1)/(T*coth(6.517*u)**2))
    dur = 1/2 * (1 + 1/np.cosh(er -1 )) * du1
    u = u + dur
    a = 1 + (1/49) * np.log((u**4 + (u/52)**2)/(u**4 + 0.432)) + (1/18.7) * np.log(1 + (u/18.1)**3)
    b = 0.564 * ((er - 0.9) / (er + 3))**0.053
    return (er + 1)/2 + (er - 1)/2 * (1 + 10/u)**(-a*b)

def _microstrip_z0(w: float, h: float, er: float, t: float):
    # compute hammerstand and jensen
    u = w/h
    T = t/h
    du1 = T/np.pi * np.log(1 + 4*np.exp(1)/(T*coth(6.517*u)**2))
    dur = 1/2 * (1 + 1/np.cosh(er -1 )) * du1
    u = u + dur
    ereff = _microstrip_ereff(w, h, er, t)
    fu = 6 + (2 * np.pi - 6) * np.exp(-((30.666/u)**0.7528))
    Z0freespace = 376.73/np.sqrt(ereff)
    return Z0freespace / (2*np.pi) * np.log(fu/u + np.sqrt(1 + (2/u)**2))

def _stripline_ereff(w, h, er, t):
    return er

def _stripline_z0(w, h, er, t):
    Z0 = 60/np.sqrt(er) * np.log((4*(2*h + t))/(0.67*np.pi *(0.8*w + t)))
    return Z0

def _w_from_z0_microstrip(targetZ0: float, h: float, er: float, t: float):
    # Find the width of the microstrip line that gives the target impedance
    # Use _microstrip_z0 to find the impedance of the line by inverting the function using interpolation
    # Use the scipy.optimize.root_scalar function to find the root of the function
    # Return the width of the line

    def f(w):
        return _microstrip_z0(w, h, er, t) - targetZ0
    
    return root_scalar(f, bracket=[0.00001, 20], xtol=0.000001, method='brentq').root

def _w_from_z0_stripline(targetZ0: float, h: float, er: float, t: float):
    # Find the width of the stripline that gives the target impedance
    # Use _stripline_z0 to find the impedance of the line by inverting the function using interpolation
    # Use the scipy.optimize.root_scalar function to find the root of the function
    # Return the width of the line

    def f(w):
        return _stripline_z0(w, h, er, t) - targetZ0
    
    return root_scalar(f, bracket=[0.00001, 20], xtol=0.000001, method='brentq').root

def _stripline_wz0(height: float, er: float, Z0: float = None, width: float = None) -> tuple[float, float]:
    # Returns both the width and impedance if one is defined
    # Raises an error if none are defined
    if Z0 is None and width is None:
        raise ValueError("Either the width or the impedance must be defined")
    if Z0 is not None:
        width = _w_from_z0_stripline(Z0, height, er)
    if width is not None:
        Z0 = _stripline_z0(width, height, er)
    return width, Z0

def _microstrip_wz0(height: float, er: float, Z0: float = None, width: float = None) -> tuple[float, float]:
    # Returns both the width and impedance if one is defined
    # Raises an error if none are defined
    if Z0 is None and width is None:
        raise ValueError("Either the width or the impedance must be defined")
    if Z0 is not None:
        width = _w_from_z0_microstrip(Z0, height, er)
    if width is not None:
        Z0 = _microstrip_z0(width, height, er)
    return width, Z0

class MicrostripLine(BaseComponent):

    def __init__(self, width: float, length: float, height: float, epsilon_r: float, tand: float, trace_thickenss: float = 60e-6, gnd: Node = None):
        super().__init__()
        self.width: float = width
        self.length: float = length
        self.height: float = height
        self.ereff: float = _microstrip_ereff(width, height, epsilon_r, trace_thickenss)
        self.Z0: float = _microstrip_z0(width, height, epsilon_r, trace_thickenss)
        self.tand: float = tand
        self.gnd: Node = gnd
        self.n_nodes = 2
        

    def __on_connect__(self):
        if self.gnd is None:
            self.gnd = self.network.gnd
        self.network.transmissionline(self.gnd, self.node(1), self.node(2),  self.Z0, self.ereff*(1-1j*self.tand), self.length)

class Stripline(BaseComponent):

    def __init__(self, width: float, length: float, height: float, epsilon_r: float, tand: float, trace_thickenss: float = 60e-6, gnd: Node = None):
        super().__init__()
        self.width: float = width
        self.length: float = length
        self.height: float = height
        self.ereff: float = _stripline_ereff(width, height, epsilon_r, trace_thickenss)
        self.Z0: float = _stripline_z0(width, height, epsilon_r, trace_thickenss)
        self.tand: float = tand
        self.gnd: Node = gnd
        self.n_nodes = 2

    def __on_connect__(self):
        if self.gnd is None:
            self.gnd = self.network.gnd
        self.network.transmissionline(self.gnd, self.node(1), self.node(2),  self.Z0, self.ereff*(1-1j*self.tand), self.length)

class Path:

    def __init__(self, start_node: Node):
        self.start_node = start_node
        self.component_link: list = []
    
    @property
    def last(self):
        return self.component_link[-1]

    @property
    def first(self):
        return len(self.component_link)


class PCBStack:

    def __init__(self, network: Network, epsilon_r: float, tand: float, thickness: float, Nlayers: int):

        # Test that the layers are at least 2 and at most 10
        if Nlayers < 2 or Nlayers > 10:
            raise ValueError("The number of layers must be at least 2 and at most 10")
        
        # Test that the thickness is positive
        if thickness <= 0:
            raise ValueError("The thickness must be positive")
        
        # Test that the relative permittivity is positive
        if epsilon_r <= 0:
            raise ValueError("The relative permittivity must be positive")
        
        # Test that the loss tangent is between 0 and 1
        if tand < 0 or tand > 1:
            raise ValueError("The loss tangent must be between 0 and 1")
        
        self.network: Network = network
        self.epsilon_r: float = epsilon_r
        self.tand: float = tand
        self.thickness: float = thickness
        self.Nlayers: int = Nlayers
        self.trace_thickness: float = 60e-6
    
    @property
    def th(self) -> float:
        return self.thickness/(self.Nlayers-1)
    
    def get_ereff(self, Z0: float, layer: int) -> float:
        if self.is_microstrip(layer):
            width = _w_from_z0_microstrip(Z0, self.th, self.epsilon_r, self.trace_thickness)
            return _microstrip_ereff(width, self.th, self.epsilon_r, self.trace_thickness)
        else:
            width = _w_from_z0_stripline(Z0, self.th, self.epsilon_r, self.trace_thickness)
            return _stripline_ereff(width, self.th, self.epsilon_r, self.trace_thickness)
        
    def is_microstrip(self, layer: int) -> bool:
        return layer == 0 or layer == self.Nlayers - 1
    
    def port(self, impedance: float, layer: int = 0) -> NodeForwarder:
        """ Create a new path starting from a port. """
        n = self.network.new_port(impedance)
        return NodeForwarder(n, layer, self, impedance)

    def transmission_line(self, n1: Node, n2: Node, length: float, impedance: float = None, width: float = None, layer: int = 0, gnd: Node = None) -> NodeForwarder:
        if layer == 0 or layer == self.Nlayers - 1:
            tl = self.microstrip_line(n1, n2, length, impedance=impedance, width=width, gnd=gnd)
        else:
            tl = self.stripline(n1, n2, length, impedance=impedance, width=width, gnd=gnd)

        Z0 = tl.Z0
        if impedance is not None:
            Z0 = impedance
        return NodeForwarder(n2, layer, self, Z0)
    
    def microstrip_line(self, n1: Node, n2: Node, length: float, impedance: float = None, width: float = None, gnd: Node = None) -> MicrostripLine:
        if gnd is None:
            gnd = self.network.gnd
        if impedance is not None and width is None:
            width = _w_from_z0_microstrip(impedance, self.th, self.epsilon_r, self.trace_thickness)
        logger.info(f'Creating microstrip line with width {width} and impedance {impedance}')
        return MicrostripLine(width, length, self.th, self.epsilon_r, self.tand, gnd=gnd).connect(n1, n2)

    def stripline(self, n1: Node, n2: Node, length: float, impedance: float = None, width: float = None, gnd: Node = None) -> Stripline:
        if gnd is None:
            gnd = self.network.gnd
        if impedance is not None:
            width = _w_from_z0_stripline(impedance, self.th, self.epsilon_r, self.trace_thickness)
        return Stripline(width, length, self.th, self.epsilon_r, self.tand, gnd=gnd).connect(n1, n2)
    
class NodeForwarder:

    def __init__(self, node: Node, layer: int, pcb: PCBStack, impedance: float):
        self.node: Node = node
        self.layer: int = layer
        self.pcb: PCBStack = pcb
        self.impedance: float = impedance

    def terminate(self, Z0: float) -> Node:
        """ Terminate the path with a resistor. """
        return self.pcb.network.terminal(self.node, Z0).output_node
    
    def straight(self, L: float, Z0: float = None, width: float = None) -> NodeForwarder:
        """ Create a straight path. """
        if Z0 is None and width is None:
            Z0 = self.impedance
        newnode = self.pcb.network.node()
        return self.pcb.transmission_line(self.node, newnode, L, impedance=Z0, width=width, layer=self.layer)
    
    def capacitor(self, C: float, node: Node = None) -> NodeForwarder:
        """ Create a capacitor. """
        if node is None:
            node = self.pcb.network.node()
            self.pcb.network.capacitor(self.node, node, C)
            return NodeForwarder(node, self.layer, self.pcb, self.impedance)
        else:
            self.pcb.network.capacitor(self.node, node, C)
            return self
        
    def inductor(self, L: float, node: Node = None) -> NodeForwarder:
        """ Create an inductor. """
        if node is None:
            node = self.pcb.network.node()
            self.pcb.network.inductor(self.node, node, L)
            return NodeForwarder(node, self.layer, self.pcb, self.impedance)
        else:
            self.pcb.network.inductor(self.node, node, L)
            return self
        
    def component(self, comp: BaseComponent) -> NodeForwarder | MultiNodeForwarder:
        """ Connect a component to the network. """
        new_nodes = [self.pcb.network.node() for _ in range(comp.n_nodes - 1)]
        comp.connect(self.node, *new_nodes)

        return MultiNodeForwarder([NodeForwarder(node, self.layer, self.pcb, self.impedance) for node in new_nodes])
    
    def via(self, newlayer: int, radius: float = 0.5e-3) -> NodeForwarder:
        """ Create a via to another layer. """
        newnode = self.pcb.network.node()
        n_layer_jump = abs(newlayer - self.layer)
        h = self.pcb.th * n_layer_jump
        r = radius
        ## Use approximal formula for via inductance
        via_inductance = u0/(2*np.pi) *(h*np.log((h+np.sqrt(r**2+h**2))/r))+3/2*(r-np.sqrt(r-np.sqrt(r**2+h**2)))
        self.pcb.network.inductor(self.node, newnode, via_inductance)
        return NodeForwarder(newnode, newlayer, self.pcb, self.impedance)
    
    def Z0_step(self, new_Z0: float) -> NodeForwarder:
        Z1 = self.impedance
        Z2 = new_Z0
        er = self.pcb.epsilon_r
        h = self.pcb.th
        if self.pcb.is_microstrip(self.layer):
            ereff1 = _microstrip_ereff(Z1, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
            w1 = _w_from_z0_microstrip(Z1, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
            ereff2 = _microstrip_ereff(Z2, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
            w2 = _w_from_z0_microstrip(Z2, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
        else:
            ereff1 = _stripline_ereff(Z1, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
            w1 = _w_from_z0_stripline(Z1, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
            ereff2 = _stripline_ereff(Z2, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
            w2 = _w_from_z0_stripline(Z2, self.pcb.th, self.pcb.epsilon_r, self.pcb.trace_thickness)
        
        if w2/w1 < 3.5:
            Cp = (np.sqrt(w1*w2)*(10.1*np.log(er) + 2.33)*w2/w1 - 12.6*np.log(er) - 3.17)*1e-12
        else:
            Cp = (130*np.log(w2/w1)-44)*1e-12

        Lp = h*(40.5*(w1/w2-1.0) - 75*w1/w2 + 0.2*((w1/w2) - 1.0)**2)*1e-9

        n1 = self.pcb.network.node()
        n2 = self.pcb.network.node()

        self.pcb.network.inductor(self.node, n1, Lp)
        self.pcb.network.capacitor(n1, self.pcb.network.gnd, Cp)
        self.pcb.network.inductor(n1, n2, Lp)
        logger.info(f'Adding parasitic inductor {Lp} nH and capacitor {Cp} pF to the Z0 step')

        return NodeForwarder(n2, self.layer, self.pcb, new_Z0)
    
    def right_turn(self, Z0: float = None, miter: bool = False) -> NodeForwarder:
        """ Create a right turn. """
        if Z0 is None:
            Z0 = self.impedance

        n1 = self.pcb.network.node()
        n2 = self.pcb.network.node()
        if self.pcb.is_microstrip(self.layer):
            ereff = _microstrip_ereff(Z0, self.pcb.th, self.pcb.epsilon_r, self.pcb.tand)
            w = _w_from_z0_microstrip(Z0, self.pcb.th, self.pcb.epsilon_r, self.pcb.tand)
        else:
            ereff = _stripline_ereff(Z0, self.pcb.th, self.pcb.epsilon_r, self.pcb.tand)
            w = _w_from_z0_stripline(Z0, self.pcb.th, self.pcb.epsilon_r, self.pcb.tand)
        h = self.pcb.th
        if w/h < 1:
            Cb = w*1e-12*((14*ereff + 12.5)*w/h - (1.83*ereff -2.25))/np.sqrt(w/h)
        else:
            Cb = w((9.5*ereff + 1.25)*(w/h)+5.2*ereff + 7.0)*1e-12
        Lb = h*1e-9*100*np.clip((4*np.sqrt(w/h)-4.21),a_min=0.001, a_max=None)
        logger.info(f'Adding parasitic inductor {Lb} nH and capacitor {Cb} pF to the right turn')
        if not miter:
            self.pcb.network.capacitor(self.node, self.pcb.network.gnd, Cb)
        self.pcb.network.inductor(self.node, n1, Lb)
        self.pcb.network.inductor(n1, n2, Lb)
        return NodeForwarder(n1, self.layer, self.pcb, Z0)
    
    def left_turn(self, Z0: float = None, miter: bool = False) -> NodeForwarder:
        return self.right_turn(Z0)
    
    def chebychev_transformer(self, Ztarget: float, fc: float, Nsections: int = 1, ripple: float = 0.05) -> NodeForwarder:
        from heavi.filtering import impedance_transformer_cheb
        Z1 = self.impedance
        Z2 = Ztarget

        Z0s = impedance_transformer_cheb(Z1, Z2, ripple, Nsections)
        logger.info(f'Adding {Nsections} Chebychev sections with ripple {ripple} to the path with impedances: {Z0s}')
        l0 = 299792458/fc * 0.25

        lastnode = self
        for Z1, Z2 in zip(Z0s[:-1], Z0s[1:-1]):
            L = l0 * 1/(np.sqrt(self.pcb.get_ereff(Z2, self.layer)))
            logger.info(f'Jump from {Z1} to {Z2} with length {L}') 
            lastnode = lastnode.Z0_step(Z2).straight(L,Z2)
        return lastnode.Z0_step(Z0s[-1])
            
class MultiNodeForwarder:

    def __init__(self, nfs: list[NodeForwarder]):
        self.nfs: list[NodeForwarder] = nfs

    def __call__(self, index: int) -> NodeForwarder:
        # check the integer range
        if index < 0 or index >= len(self.nfs):
            raise ValueError("The index must be between 0 and the number of nodes - 1")
        return self.nfs[index]
    