from __future__ import annotations
from enum import Enum
from typing import List, Callable
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from numba import njit, prange, c16, i8, f8
import numba_progress as nbp
from numba_progress.progress import ProgressBarType

from loguru import logger

from .sparam import Sparameters
from .numeric import Scalar, parse_numeric, Function

TEN_POWERS = {
    -12: "p",
    -9: "n",
    -6: "u",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "T",
    12: "P",
}

def _get_power(number: float):
    tp = np.log10(np.abs(number))
    v = np.floor(tp / 3) * 3
    v = min(12, max(-12, v))
    v2 = number / (10**v)
    return v2, v

def Z0_VSWR(Z0: float, max_vswr: float) -> float:
    """
    Returns a random real impedance corresponding to a random VSWR between 1
    and max_vswr relative to the reference impedance Z0.

    Parameters
    ----------
    Z0 : float
        The reference impedance (Ohms).
    max_vswr : float
        The maximum possible VSWR.

    Returns
    -------
    impedance : float
        A random real impedance corresponding to the random VSWR.
    """
    # Generate a random VSWR uniformly distributed between 1 and max_vswr
    vswr = np.random.uniform(1, max_vswr)
    
    # Compute the corresponding reflection coefficient magnitude (rho)
    rho = (vswr - 1) / (vswr + 1)
    
    # Map rho to an impedance value using the formula for real impedances:
    # Z = Z0 * (1 + rho) / (1 - rho)
    impedance = Z0 * (1 + rho) / (1 - rho)
    
    return impedance


class ComponentType(Enum):
    UNDEFINED = 0
    RESISTOR = 1
    CAPACITOR = 2
    INDUCTOR = 3
    CURRENTSOURCE = 4
    VOLTAGESOURCE = 5
    IMPEDANCE = 6
    ADMITTANCE = 7
    TRANSMISSIONLINE = 8
    NPORT = 9
    CUSTOM = 10

    @property
    def unit(self):
        unitdict = {
            self.UNDEFINED: "__",
            self.RESISTOR: "Ohm",
            self.CAPACITOR: "F",
            self.INDUCTOR: "H",
            self.CURRENTSOURCE: "A",
            self.VOLTAGESOURCE: "V",
            self.IMPEDANCE: "Ohm",
            self.ADMITTANCE: "G",
            self.TRANSMISSIONLINE: "Ohm",
            self.NPORT: "None",
        }
        return unitdict[self]


twopi = 2 * np.pi
pi = np.pi


def randphase():
    """ Returns a complex number with a random phase. """
    return np.exp(complex(0, 2 * np.pi * np.random.random_sample()))


def randmag(minv, maxv):
    """ Returns a random number between minv and maxv. """
    return (maxv - minv) * np.random.random_sample() + minv


def randomphasor(minv=0, maxv=1):
    """ Returns a random complex number with a random phase and magnitude. """
    return randmag(minv, maxv) * randphase()


def _make_callable(value: float | complex | Callable) -> Callable:
    """ Returns a callable function for a given value. """
    if isinstance(value, Callable):
        return value
    else:
        return lambda f: value*np.ones_like(f)
    
@njit(cache=True, parallel=True, fastmath=True)
def solve_single_frequency_c_compiled(Is, Ys, Zs, indices, frequencies, progprox):
    nT = len(indices)
    nF = len(frequencies)
    Ss = np.zeros((nT,nT,nF), dtype=np.complex128)
    Vdonor = np.zeros((Is.shape[0],), dtype=np.complex128)
    for it in range(nT):
        ind1 = indices[it]
        for i in prange(nF):
            Vh = 0*Vdonor
            Vh[1:] = np.linalg.solve(Ys[:,:,i],Is[1:,it,i])
            for itt in range(nT):
                ind2 = indices[itt]
                Q = np.sqrt(
                    np.abs(np.real(Zs[it,i]))
                ) / np.sqrt(np.abs(np.real(Zs[itt,i])))
                Ss[itt, it, i] = Q * (
                    (Vh[ind2] * 2 - Zs[itt,i]* Is[ind2,it,i])
                    / (Zs[it,i] * Is[ind1,it,i])
                )
            progprox.update(1)
    return Ss

@njit(cache=True, parallel=True, fastmath=True)
def compute_s_parameters(Is, Ys, Zs, port_indices, frequencies, progress_object):
    """
    Compute the S-parameter matrix for an RF network using Numba for acceleration.

    Parameters
    ----------
    Is : numpy.ndarray
        Current sources, complex-valued array of shape (n_nodes, n_ports, n_freqs).
    Ys : numpy.ndarray
        Admittance matrices, complex-valued array of shape (n_nodes, n_nodes, n_freqs).
    Zs : numpy.ndarray
        Source impedances, complex-valued array of shape (n_nodes, n_freqs).
    port_indices : numpy.ndarray
        Indices of the nodes corresponding to the ports of interest, integer array of shape (n_ports,).
    frequencies : numpy.ndarray
        Frequencies, float-valued array of shape (n_freqs,).

    Returns
    -------
    S_parameters : numpy.ndarray
        S-parameter matrix, complex-valued array of shape (n_ports, n_ports, n_freqs).
    """
    num_ports = len(port_indices)
    num_freqs = len(frequencies)
    num_nodes = Is.shape[0]

    # Initialize the S-parameter matrix
    S_parameters = np.zeros((num_ports, num_ports, num_freqs), dtype=np.complex128)

    # Voltage vector placeholder
    Vh = np.zeros((num_nodes,), dtype=np.complex128)

    for port_in_idx in range(num_ports):
        node_in = port_indices[port_in_idx]
        for freq_idx in prange(num_freqs):
            # Reset voltage vector
            Vh = 0*Vh

            # Solve the system of equations for Vh[1:]
            Vh[1:] = np.linalg.solve(
                Ys[:,:, freq_idx],
                Is[1:, port_in_idx, freq_idx]
            )

            Z_in = Zs[port_in_idx, freq_idx]

            for port_out_idx in range(num_ports):
                node_out = port_indices[port_out_idx]
                Z_out = Zs[port_out_idx, freq_idx]

                # Calculate scaling factor Q
                Q = np.sqrt(np.abs(np.real(Z_in))) / np.sqrt(np.abs(np.real(Z_out)))

                # Compute numerator and denominator for S-parameter calculation
                numerator = Vh[node_out] * 2 - Z_out * Is[node_out, port_in_idx, freq_idx]
                denominator = Z_in * Is[node_in, port_in_idx, freq_idx]

                # Compute S-parameter
                S_parameters[port_out_idx, port_in_idx, freq_idx] = Q * (numerator / denominator)
            progress_object.update(1)
    return S_parameters

@njit(cache=True, parallel=True, fastmath=True)
def compute_s_parameters_no_loadbar(Is, Ys, Zs, port_indices, frequencies):
    """
    Compute the S-parameter matrix for an RF network using Numba for acceleration.

    Parameters
    ----------
    Is : numpy.ndarray
        Current sources, complex-valued array of shape (n_nodes, n_ports, n_freqs).
    Ys : numpy.ndarray
        Admittance matrices, complex-valued array of shape (n_nodes, n_nodes, n_freqs).
    Zs : numpy.ndarray
        Source impedances, complex-valued array of shape (n_nodes, n_freqs).
    port_indices : numpy.ndarray
        Indices of the nodes corresponding to the ports of interest, integer array of shape (n_ports,).
    frequencies : numpy.ndarray
        Frequencies, float-valued array of shape (n_freqs,).

    Returns
    -------
    S_parameters : numpy.ndarray
        S-parameter matrix, complex-valued array of shape (n_ports, n_ports, n_freqs).
    """
    num_ports = len(port_indices)
    num_freqs = len(frequencies)
    num_nodes = Is.shape[0]

    # Initialize the S-parameter matrix
    S_parameters = np.zeros((num_ports, num_ports, num_freqs), dtype=np.complex128)

    # Voltage vector placeholder
    Vh = np.zeros((num_nodes,), dtype=np.complex128)

    for port_in_idx in range(num_ports):
        node_in = port_indices[port_in_idx]
        for freq_idx in prange(num_freqs):
            # Reset voltage vector
            Vh = 0*Vh

            # Solve the system of equations for Vh[1:]
            Vh[1:] = np.linalg.solve(
                Ys[:,:, freq_idx],
                Is[1:, port_in_idx, freq_idx]
            )

            Z_in = Zs[port_in_idx, freq_idx]

            for port_out_idx in range(num_ports):
                node_out = port_indices[port_out_idx]
                Z_out = Zs[port_out_idx, freq_idx]

                # Calculate scaling factor Q
                Q = np.sqrt(np.abs(np.real(Z_in))) / np.sqrt(np.abs(np.real(Z_out)))

                # Compute numerator and denominator for S-parameter calculation
                numerator = Vh[node_out] * 2 - Z_out * Is[node_out, port_in_idx, freq_idx]
                denominator = Z_in * Is[node_in, port_in_idx, freq_idx]

                # Compute S-parameter
                S_parameters[port_out_idx, port_in_idx, freq_idx] = Q * (numerator / denominator)
    return S_parameters

@dataclass
class Node:
    """ Node class for the Network object. """
    name: str
    _index: int = None
    _parent: Network = None
    _linked: Node = None

    def __hash__(self):
        return hash(f'{self.name}_{self.index}')
    
    def set_index(self, index: int):
        self._index = index

    @property
    def index(self) -> int:
        if self._linked is not None:
            return self._linked.index
        return self._index

@dataclass
class ComponentFunction:
    """ ComponentFunction class for the Component object. """
    node_list: list[Node]
    simval: Scalar
    as_matrix: bool = False

    @property
    def matrix_slice(self) -> tuple:
        """ Returns a tuple of slices for the matrix indices corresponding to the nodes. """
        if self.as_matrix:
            idx = [node.index for node in self.node_list]
            return np.ix_(idx,idx) + (slice(None),)
        if len(self.node_list) == 1:
            return (self.node_list[0].index, slice(None))
        elif len(self.node_list) == 2:
            return (self.node_list[0].index, self.node_list[1].index, slice(None))

        
class Component:
    """ Component class for the Network object. 
    This class represents a component in the network, such as a resistor, capacitor, inductor, etc.
    
    Parameters
    ----------
    nodes : list[Node]
        A list of Node objects corresponding to the nodes the component is connected to.
    functionlist : list[ComponentFunction]
        A list of ComponentFunction objects corresponding to the functions of the component.
    type : ComponentType
        The type of the component.
    display_value : float
        The value of the component.

    """
    def __init__(
        self, nodes, functionlist, type: ComponentType = ComponentType.UNDEFINED, component_value: Scalar = None
    ):
        self.nodes: list[Node] = nodes
        self.functionlist: list[ComponentFunction] = functionlist
        self.type: ComponentType = type
        self._component_value: Scalar = parse_numeric(component_value)
        self._impedance: Scalar = None

    @property
    def _display_value(self) -> float:
        return self._component_value.value
    
    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        value, power = _get_power(self._display_value)
        return f"{self.type.name}: {[s.name for s in self.nodes]}, value={value:.2f} {TEN_POWERS[power]}{self.type.unit}"
        

    def _generate_compiler_function(self) -> Callable:
        '''Generates a callable that will plug in the components matrix entries for a given frequency.'''
        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            for function in self.functionlist:
                matrix[function.matrix_slice] += function.simval(f)
            return matrix
        return compiler

@dataclass
class Terminal:
    ground: Node
    port: Node
    source: Component
    source_impedance: Component

    @property
    def ground_node(self) -> Node:
        return self.source.nodes

    @property
    def z_source(self) -> Scalar:
        return self.source_impedance._impedance


class Network:
    """ Network class for the Network object.
    This class represents a network of components and nodes. It is used to build and analyze circuits.
    
    Parameters
    ----------
    default_name : str
        The default name for a node.
    node_name_counter_start : int
        The starting index for the node name counter.
    """

    def __init__(self, default_name: str = 'Node', suppress_loadbar: bool = False):
        self.gnd: Node = Node("gnd", _parent=self)
        self.nodes: list[Node] = [self.gnd]
        self.components: list[Component] = []
        self.sources: list[Component] = []
        self.terminals: list[Terminal] = []
        self.node_counter: defaultdict[str, int] = defaultdict(int)
        self.node_default_name: str = default_name
        self.suppress_loadbar: bool = suppress_loadbar
        

    def print_components(self) -> None:
        '''Prints an overview of the components in the Network'''
        for comp in self.components + self.sources:
            logger.info(comp)

    @property
    def node_names(self) -> list[str]:
        '''A list of strings corresponding to each node.'''
        return [n.name for n in self.nodes]

    def get_node_index(self, node: str) -> int:
        '''Returns the index of a node corresponding with the tag name.'''
        return self.nodes.index(node)

    def _compile_nodes(self) -> None:
        '''_compile_nodes writes an index number to the node's index field required for matrix lookup.'''
        for i, node in enumerate(self.nodes):
            node.set_index(i)

    def _new_node_name(self, basename: str = None) -> str:
        '''Generates a node name label to be used by checking which ones exist and then generating a new one.'''
        if basename is None:
            basename = self.node_default_name
        node_name = f'{basename}{self.node_counter[basename]}'
        self.node_counter[basename] += 1
        return node_name
    
    def named_node(self, prefix: str) -> Node:
        """ Adds a named node to the network where the prefix is predetermined. """
        name = self._new_node_name(prefix)
        
        N = Node(name, _parent=self)
        self.nodes.append(N)
        return N
    
    def node(self, name: str = None) -> Node:
        '''Generates a new Node object for a node with the optionally provided label. Returns a Node object.'''
        if name is None:
            name = self._new_node_name()
        
        if name in self.node_names:
            logger.warning(f"Node name {name} already exists")
            name = self._new_node_name(name)
    
        N = Node(name, _parent=self)
        self.nodes.append(N)
        return N

    def mnodes(self, N: int, name: str = None) -> list[Node]:
        if name is None:
            return [self.node() for _ in range(N)]
        else:
            return [self.node(f'{name}_{i+1}') for i in range(N)]
    
    def _check_unconnected_nodes(self) -> None:
        '''Checks for unconnected nodes in the network and raises a warning if any are found.'''
        # collecct a list of nodes and included status
        node_dict = {node: False for node in self.nodes}

        # check if nodes are used
        for component in self.components:
            for node in component.nodes:
                node_dict[node] = True
        
        # check if nodes are used
        for terminal in self.terminals:
            for node in terminal.source.nodes:
                node_dict[node] = True
        
        # check if nodes are used

        for node, used in node_dict.items():
            if not used:
                logger.error(f"Node {node.name} is not connected to any components.")
                logger.error("Unconnected nodes will cause the analysis to yield 0 values.")
                raise ValueError(f"Node {node.name} is not connected to any components.")
        
    def run_sparameter_analysis(self, frequencies: np.ndarray) -> Sparameters:
        logger.warning("run_sparameter_analysis will be deprecated. Use run instead.")
        return self.run(frequencies)

    def run(self, frequencies: np.ndarray) -> Sparameters:
        """
        Runs an S-parameter analysis for the network at the specified frequencies.

        Parameters:
        -----------
        frequencies (np.ndarray): An array of frequencies at which to run the analysis.

        Returns:
        --------
        Sparameters: An Sparameters object containing the S-parameter matrix for the network at the specified frequencies

        """
        self._compile_nodes()

        self._check_unconnected_nodes()
        nT = len(self.terminals)
        nF = len(frequencies)
        nV = len(self.nodes)

        component_compilers = [c._generate_compiler_function() for c in self.components]
        terminal_compilers = [t.source._generate_compiler_function() for t in self.terminals]

        ntot = nT * nF

        Y = np.zeros((nV, nV, nF), dtype=np.complex128)

        for compiler in component_compilers:
            Y = compiler(Y, frequencies)

        Ys = Y[1:, 1:, :]
        Zs = np.zeros((nT,nF), dtype=np.complex128)
        Is = np.zeros((nV,nT,nF), dtype=np.complex128)
        for it in range(nT):
            Is[:,it,:] = terminal_compilers[it](Is[:,it,:],frequencies)
            Zs[it,:] = self.terminals[it].z_source(frequencies)

        indices = np.array([self.get_node_index(self.terminals[ii].port) for ii, _ in enumerate(self.terminals)]).astype(np.int32)
        frequencies = np.array(frequencies).astype(np.float32)
        Sol = None

        if self.suppress_loadbar:
            Sol = compute_s_parameters_no_loadbar(Is, Ys, Zs, indices, frequencies)
        else:
            with nbp.ProgressBar(total=ntot) as progress:
                Sol = compute_s_parameters(Is, Ys, Zs, indices, frequencies, progress) 
        
        return Sparameters(Sol, frequencies)
    

    def current_source(self, node_from: Node, node_to: Node, current: float | Scalar) -> Component:
        """
        Adds a current source between two nodes in the circuit.

        Parameters:
        -----------
        node_from (Node): The node from which the current source originates.
        node_to (Node): The node to which the current source is connected.
        current (float): The current value of the current source in Amperes.

        """
        functionlist = []
        current = parse_numeric(current)
        functionlist.append(ComponentFunction([node_from], current.negative()))
        functionlist.append(ComponentFunction([node_to], current))
        current_source_obj = Component(
            [node_from, node_to], functionlist, type=ComponentType.CURRENTSOURCE, component_value=current
        )
        self.sources.append(current_source_obj)
        return current_source_obj

    def terminal(self, signal_node: Node, Z0: float | Scalar, gnd_node: Node = None) -> Terminal:
        """ Adds a terminal to the network and returns the created terminal object.
        Parameters:
        -----------
        signal_node (Node): The node to which the terminal is connected.
        Z0 (float): The characteristic impedance of the terminal.
        gnd_node (Node, optional): The ground node of the terminal. Defaults to network.gnd.

        Returns:
        --------
        Terminal: The created terminal object.
        
        """

        if gnd_node is None:
            gnd_node = self.gnd
        Z0 = parse_numeric(Z0)
        impedance_component = self.impedance(gnd_node, signal_node, Z0, display_value=Z0)
        current_source = self.current_source(gnd_node, signal_node, Z0.inverse())

        terminal_object = Terminal(gnd_node, signal_node, current_source, impedance_component)
        self.terminals.append(terminal_object)
        return terminal_object

    def port(self, impedance: float) -> Node:
        '''Returns a tuple containing a Node and Terminal object.
        The Node object is generated with a name corresponding to the number.
        The Terminal object is generated with the Node object and the provided impedance
        
        Parameters:
        -----------
        impedance (float): The impedance value for the Terminal object.
        
        Returns:
        --------
        Node: The ports output node
        '''

        node = self.node()
        self.terminal(node, impedance)
        return node
    
    def admittance(self, node1: Node, node2: Node, Y: float, 
                  component_type: ComponentType = ComponentType.ADMITTANCE,
                  display_value: float = None) -> Component:
        """
        Adds an admittance component between two nodes and returns the created component.
        Parameters:
        -----------
            node1 (Node): The first node of the admittance component.
            node2 (Node): The second node of the admittance component.
            Y (float): The admittance value of the component.
        
        Returns:
        --------
            Component: The created admittance component.
        """
        
        functionlist = []
        
        admittance_simvalue = parse_numeric(Y)

        if display_value is None:
            display_value = admittance_simvalue
            logger.debug(f'Defaulting display value to {display_value}')

        functionlist.append(ComponentFunction([node1, node1], admittance_simvalue))
        functionlist.append(ComponentFunction([node1, node2], admittance_simvalue.negative()))
        functionlist.append(ComponentFunction([node2, node1], admittance_simvalue.negative()))
        functionlist.append(ComponentFunction([node2, node2], admittance_simvalue))

        admittance_component = Component([node1, node2], functionlist, type=component_type, component_value=display_value)
        admittance_component._impedance = admittance_simvalue.inverse()
        self.components.append(admittance_component)
        return admittance_component

    def impedance(self, node1: Node, node2: Node, Z: float, 
                  component_type: ComponentType = ComponentType.IMPEDANCE,
                  display_value: float = None) -> Component:
        """Creates and returns a component object for an impedance.

        Parameters:
        -----------
        node1 (Node): The first node of the impedance.
        node2 (Node): The second node of the impedance.
        Z (float): The impedance value of the impedance in ohms.
        component_type (ComponentType, optional): The type of the component. Defaults to ComponentType.IMPEDANCE.
        display_value (float, optional): The value to display for the component. Defaults to None.

        Returns:
        --------
        Component: The created impedance component object.

        """
        functionlist = []
        
        impedance = parse_numeric(Z)
        admittance = parse_numeric(Z, inverse=True)
        
        if display_value is None:
            display_value = impedance(1)
            logger.debug(f'Defaulting display value to {display_value}')
        
        functionlist.append(ComponentFunction([node1, node1], admittance))
        functionlist.append(ComponentFunction([node1, node2], admittance.negative()))
        functionlist.append(ComponentFunction([node2, node1], admittance.negative()))
        functionlist.append(ComponentFunction([node2, node2], admittance))
        impedance_object = Component([node1, node2], functionlist, type=component_type, component_value=display_value)
        impedance_object._impedance = impedance
        self.components.append(impedance_object)
        return impedance_object

    def resistor(self, node1: Node, node2: Node, R: float):
        """
        Adds a resistor between two nodes in the circuit.

        Parameters:
        -----------
            node1 (Node): The first node to which the resistor is connected.
            node2 (Node): The second node to which the resistor is connected.
            R (float): The resistance value of the resistor in ohms.

        Returns:
        --------
        Impedance: The impedance object representing the resistor between the two nodes.
        """
        
        return self.impedance(node1, node2, R, component_type=ComponentType.RESISTOR, display_value=R)

    def capacitor(self, node1: Node, node2: Node, C: float) -> Component:
        """
        Creates and returns a component object for a capacitor.

        Parameters:
        -----------

        node1 (Node): The first node of the capacitor.
        node2 (Node): The second node of the capacitor.
        C (float): The capacitance value of the capacitor in Farads.

        Returns:
        --------
        Component: The created capacitor component object.

        """
        C = parse_numeric(C)
        
        def admittance_f(f):
            return 1j * twopi * f * C(f)
        
        admittance = Function(admittance_f)
        
        return self.admittance(node1, node2, admittance, component_type=ComponentType.CAPACITOR, display_value=C)
        
        
    def inductor(self, node1: Node, node2: Node, L: float):
        """
        Adds an inductor component between two nodes in the circuit.
        Args:
            node1 (Node): The first node to which the inductor is connected.
            node2 (Node): The second node to which the inductor is connected.
            L (float): The inductance value of the inductor in Henrys.
        Returns:
            Component: The created inductor component.
        """
        L = parse_numeric(L)
        
        def admittance_f(f):
            return 1/(1j * twopi * f * L(f))
        
        admittance = Function(admittance_f)
        
        return self.admittance(node1, node2, admittance, component_type=ComponentType.INDUCTOR, display_value=L)
    
        
    def transmissionline(
        self, gnd: Node, port1: Node, port2: Node, Z0: float, er: float, L: float
    ) -> Component:
        """
        Creates and returns a component object for a transmission line.

        Parameters:
        -----------
        gnd (Node): The ground node.
        port1 (Node): The first port node.
        port2 (Node): The second port node.
        Z0 (float): Characteristic impedance of the transmission line.
        er (float): Relative permittivity of the transmission line.
        L (float): Length of the transmission line.

        Returns:
        --------
        Component: A component object representing the transmission line.
        """
        functionlist = []
        c0 = 299792458
        func_er = parse_numeric(er)
        func_Z0 = parse_numeric(Z0)
        func_L = parse_numeric(L)

    
        functionlist.append(
            ComponentFunction(
                [gnd, gnd],
                Function(lambda f: 1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0)))
                ,
            )
        )
        functionlist.append(
            ComponentFunction(
                [gnd, port1],
                Function(lambda f: -(
                    1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                    - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                )),
            )
        )
        functionlist.append(
            ComponentFunction(
                [port1, gnd],
                Function(lambda f: -(
                    1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                    - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                )),
            )
        )
        functionlist.append(
            ComponentFunction(
                [port1, port1],
                Function(lambda f: 1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))),
            )
        )

        functionlist.append(
            ComponentFunction(
                [port1, port1],
                Function(lambda f: 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))),
            )
        )
        functionlist.append(
            ComponentFunction(
                [port1, port2],
                Function(lambda f: -1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))),
            )
        )
        functionlist.append(
            ComponentFunction(
                [port2, port1],
                Function(lambda f: -1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))),
            )
        )
        functionlist.append(
            ComponentFunction(
                [port2, port2],
                Function(lambda f: 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))),
            )
        )

        functionlist.append(
            ComponentFunction(
                [gnd, gnd],
                Function(lambda f: 1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))),
            )
        )
        functionlist.append(
            ComponentFunction(
                [gnd, port2],
                Function(lambda f: -(
                    1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                    - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                )),
            )
        )
        functionlist.append(
            ComponentFunction(
                [port2, gnd],
                Function(lambda f: -(
                    1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                    - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                )),
            )
        )
        functionlist.append(
            ComponentFunction(
                [port2, port2],
                Function(lambda f: 1 / (func_Z0(f) * np.tanh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))
                - 1 / (func_Z0(f) * np.sinh(1j * func_L(f) * 2 * pi * f * np.sqrt(func_er(f)) / c0))),
            )
        )

        transmissionline_component = Component(
            [gnd, port1, port2], functionlist, type=ComponentType.TRANSMISSIONLINE, component_value=Z0
        )
        self.components.append(transmissionline_component)
        return transmissionline_component

    def TL(self, node1: Node, node2: Node, beta: float | Scalar, length: float | Scalar, Z0: float | Scalar):
        beta = parse_numeric(beta)
        length = parse_numeric(length)
        Z0 = parse_numeric(Z0)

        def a11(f):
            return np.cosh(1j*beta(f)*length(f))
        def a12(f):
            return Z0(f)*np.sinh(1j*beta(f)*length(f))
        def a21(f):
            return 1/Z0(f)*np.sinh(1j*beta(f)*length(f))
        def a22(f):
            return np.cosh(1j*beta(f)*length(f))
        
        def y11(f):
            return a22(f)/a12(f)
        def y12(f):
            return -((a11(f)*a22(f))-(a12(f)*a21(f)))/a12(f)
        def y21(f):
            return -1/a12(f)
        def y22(f):
            return a11(f)/a12(f)
        
        self.n_port_Y(self.gnd, [node1, node2], [[y11, y12], [y21, y22]], Z0, component_type=ComponentType.TRANSMISSIONLINE)

    def two_port_reciprocal(
        self,
        gnd: Node,
        port1: Node,
        port2: Node,
        S11: complex,
        S12: complex,
        S22: complex,
        Z0: float,
    ) -> tuple[Component, Component, Component]:
        """
        Calculate the admittance parameters for a two-port reciprocal network.
        Args:
            gnd (Node): The ground node.
            port1 (Node): The first port node.
            port2 (Node): The second port node.
            S11 (complex): The S11 scattering parameter as a function of frequency.
            S12 (complex): The S12 scattering parameter as a function of frequency.
            S22 (complex): The S22 scattering parameter as a function of frequency.
            Z0 (float): The characteristic impedance.
        Returns:
            tuple[Component, Component, Component]: A tuple containing the admittance components Y1, Y2, and Y3.
        """
        Z0 = parse_numeric(Z0)
        def detS(f):
            return ((1 + S11(f)) * (1 + S22(f))) - S12(f) ** 2
        def Y11(f):
            return ((1 - S11(f)) * (1 + S22(f)) + S12(f) ** 2) / (detS(f)) * 1 / Z0(f)
        def Y12(f):
            return -2 * S12(f) / detS(f) * 1 / Z0(f)
        def Y22(f):
            return ((1 + S11(f)) * (1 - S22(f)) + S12(f) ** 2) / (detS(f)) * 1 / Z0(f)

        Y1 = self.admittance(gnd, port1, lambda f: Y11(f) + Y12(f))
        Y2 = self.admittance(port1, port2, lambda f: -Y12(f))
        Y3 = self.admittance(gnd, port2, lambda f: Y22(f) + Y12(f))

        return (Y1, Y2, Y3)

    def n_port_S(
            self,
            gnd: Node,
            nodes: list[Node],
            Sparam: list[list[Callable]],
            Z0: float,
            component_type: ComponentType = ComponentType.NPORT
    ) -> None:
        """Adds an N-port S-parameter component to the circuit.

        Parameters:
        -----------
        gnd : Node
            The ground node of the circuit.
        nodes : list[Node]
            List of nodes representing the ports of the N-port network.
        Sparam : list[list[Callable]]
            A nested list of callables representing the S-parameters as functions of frequency.
        Z0 : float
            The reference impedance.
        Returns:
        --------
        None
        Notes:
        ------
        This method constructs the admittance matrix (Y-parameters) from the given S-parameters
        and adds the corresponding component to the circuit's component list.
        """
        N = len(nodes)

        def comp_function(f: float):
            nF = f.shape[0]
            S = np.array([[sp(f) for sp in row] for row in Sparam], dtype=np.complex128)
            Identity = np.repeat(np.eye(N)[:, :, np.newaxis], nF, axis=2).astype(np.complex128)
            Y = (1/Z0) * np.einsum('ijk,jlk->ilk', (Identity-S), np.stack([np.linalg.inv((Identity+S)[:, :, m]) for m in range(nF)],axis=2))
            Y2 = np.zeros((N+1,N+1,nF),dtype=np.complex128)
            Y2[:N,:N,:] = Y
            for i in range(N):
                Y2[i,N,:] = -np.sum(Y[i,:,:],axis=0)
                Y2[N,i,:] = -np.sum(Y[:,i,:],axis=0)
                Y2[N,N,:] += np.sum(Y[i,:,:],axis=0)
            return Y2
        component = Component(nodes + [gnd, ],[ComponentFunction(nodes + [gnd, ],Function(comp_function),True),],component_type, Z0 )
        self.components.append(component)

    def n_port_Y(
            self,
            gnd: Node,
            nodes: list[Node],
            Yparams: list[list[Callable]],
            Z0: float,
            component_type: ComponentType = ComponentType.NPORT
    ) -> None:
        """Adds an N-port Y-parameter component to the circuit.

        Parameters:
        -----------
        gnd : Node
            The ground node of the circuit.
        nodes : list[Node]
            List of nodes representing the ports of the N-port network.
        Yparam : list[list[Callable]]
            A nested list of callables representing the Y-parameters as functions of frequency.
        Z0 : float
            The reference impedance.
        Returns:
        --------
        None
        Notes:
        ------
        This method constructs the admittance matrix (Y-parameters)and adds the corresponding component 
        to the circuit's component list.
        """
        N = len(nodes)

        def comp_function(f: float):
            nF = f.shape[0]
            Y = np.array([[sp(f) for sp in row] for row in Yparams], dtype=np.complex128)
            Y2 = np.zeros((N+1,N+1,nF),dtype=np.complex128)
            Y2[:N,:N,:] = Y
            for i in range(N):
                Y2[i,N,:] = -np.sum(Y[i,:,:],axis=0)
                Y2[N,i,:] = -np.sum(Y[:,i,:],axis=0)
                Y2[N,N,:] += np.sum(Y[i,:,:],axis=0)
            return Y2
        component = Component(nodes + [gnd, ],[ComponentFunction(nodes + [gnd, ],Function(comp_function),True),],component_type, Z0 )
        self.components.append(component)
    
    

    # An old implementation of the transmission line function that is not used and not convenient.
    
    # def transmissionline_partwise(
    #     self, gnd: Node, port1: Node, port2: Node, func_z0: float, func_er: float, L: float
    # ) -> tuple[Component, Component, Component]:
    #     '''Generates and returns a tuple of three impedance components that correspond to a transmission line.
    #     The transmission line is divided into three parts: port1 to port2, port1 to gnd, and port2 to gnd.
        
    #     Parameters
    #     ----------
    #     gnd : Node
    #         The ground node.
    #     port1 : Node
    #         The first port node.
    #     port2 : Node
    #         The second port node.
        
    #     '''
    #     functionlist = []
    #     c0 = 299792458
    #     func_er = _make_callable(func_er)
    #     func_z0 = _make_callable(func_z0)

    #     Z1 = self.impedance(
    #         gnd,
    #         port1,
    #         lambda f: 1 / (func_z0(f) * np.tanh(L * 2 * pi * f * np.sqrt(func_er(f)) / c0))
    #         - 1 / (func_z0(f) * np.sinh(L * 2 * pi * f * np.sqrt(func_er(f)) / c0)),
    #     )
    #     Z2 = self.impedance(
    #         port1,
    #         port2,
    #         lambda f: 1 / (func_z0(f) * np.sinh(L * 2 * pi * f * np.sqrt(func_er(f)) / c0)),
    #     )
    #     Z3 = self.impedance(
    #         gnd,
    #         port2,
    #         lambda f: 1 / (func_z0(f) * np.tanh(L * 2 * pi * f * np.sqrt(func_er(f)) / c0))
    #         - 1 / (func_z0(f) * np.sinh(L * 2 * pi * f * np.sqrt(func_er(f)) / c0)),
    #     )
    #     return [Z1, Z2, Z3]
