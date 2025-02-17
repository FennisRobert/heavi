from __future__ import annotations
from enum import Enum
from typing import List, Callable
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from .solver import solve_MNA_RF, solve_MNA_RF_nopgb
import numba_progress as nbp
from loguru import logger

from .sparam import Sparameters
from .numeric import Scalar, parse_numeric, Function, SimParam

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

def format_value_units(value: float, unit: str) -> str:
    """ Formats a value with units for display. """
    
    v, p = _get_power(value)
    return f"{v:.2f} {TEN_POWERS[p]}{unit}"

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


class Components:
    RESISTOR = {'name': 'Resistor', 'unit': 'Ω'}
    CAPACITOR = {'name': 'Capacitor', 'unit': 'F'}
    INDUCTOR = {'name': 'Inductor', 'unit': 'H'}
    CURRENTSOURCE = {'name': 'Current Source', 'unit': 'A'}
    VOLTAGESOURCE = {'name': 'Voltage Source', 'unit': 'V'}
    IMPEDANCE = {'name': 'Impedance', 'unit': 'Ω'}
    ADMITTANCE = {'name': 'Admittance', 'unit': 'G'}
    TRANSMISSIONLINE = {'name': 'Transmission Line', 'unit': 'Ω'}
    NPORT = {'name': 'N-Port', 'unit': 'None'}
    CUSTOM = {'name': 'Custom', 'unit': 'None'}

#     @property
#     def unit(self):
#         unitdict = {
#             self.UNDEFINED: "__",
#             self.RESISTOR: "Ohm",
#             self.CAPACITOR: "F",
#             self.INDUCTOR: "H",
#             self.CURRENTSOURCE: "A",
#             self.VOLTAGESOURCE: "V",
#             self.IMPEDANCE: "Ohm",
#             self.ADMITTANCE: "G",
#             self.TRANSMISSIONLINE: "Ohm",
#             self.NPORT: "None",
#         }
#         return unitdict[self]


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


@dataclass
class Node:
    """ Node class for the Network object. """
    name: str
    _index: int = None
    _parent: Network = None
    _linked: Node = None
    _gnd: bool = False

    def __repr__(self) -> str:
        if self._gnd:
            return 'Node[GND]'
        if self._linked is None:
            return f"Node[{self._index}]"
        else:
            return f"LinkedNode[{self._index}>{self._linked._index}]"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __hash__(self):
        return hash(f'{self.name}_{self.index}')
    
    def set_index(self, index: int):
        self._index = index

    def unique(self) -> Node:
        if self._linked is not None:
            return self._linked
        return self
    
    @property
    def index(self) -> int:
        if self._linked is not None:
            return self._linked.index
        return self._index

    def merge(self, other: Node) -> Node:
        self._linked = other
        return self
    
    def __gt__(self, other: Node) -> Node:
        if isinstance(other, Node):
            #logger.info(f'Merged {self} with {other}')
            self._linked = other
            return other
        return NotImplemented
    

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
        self, nodes, functionlist, component_value: Scalar = None
    ):
        self.nodes: list[Node] = nodes
        self.functionlist: list[ComponentFunction] = functionlist
        self._component_value: Scalar = parse_numeric(component_value)
        self._impedance: Scalar = None
        self.meta_data: dict[str, str] = dict()

        self.component_name: str = None
        self.component_unit: str = None

    @property
    def display_value(self) -> str:
        return format_value_units(self._component_value.scalar(), self.component_unit)
    
    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        value = self.display_value
        return f"{self.component_name}: {[str(n) for n in self.nodes]}, value={value}"
        
    def set_metadata(self, name: str = 'Component', unit: str = '', value: float | SimParam = None, **kwargs: dict[str, str]) -> Component:
        self.component_name = name
        self.component_unit = unit
        if value is not None:
            self._component_value = parse_numeric(value)
        self.meta_data.update(kwargs)
        return self
    
    def _generate_compiler_function(self) -> Callable:
        '''Generates a callable that will plug in the components matrix entries for a given frequency.'''
        def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
            for function in self.functionlist:
                matrix[function.matrix_slice] += function.simval(f)
            return matrix
        return compiler

@dataclass
class Source:
    ground: Node
    node: Node
    source_node: Node
    source_impedance: Component
    dc_voltage: float = 1
    ac_voltage: float = 1

    @property
    def ground_node(self) -> Node:
        return self.ground
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
        self.gnd: Node = Node("gnd", _parent=self, _gnd=True)
        self.nodes: list[Node] = [self.gnd]
        self.components: list[Component] = []
        self.sources: list[Source] = []
        self.ports: dict[int, Source] = {}
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

    def unlinked_nodes(self) -> list[Node]:
        '''Returns a list of nodes that are not linked to any other nodes.'''
        return [node for node in self.nodes if node._linked is None]
    
    def _compile_nodes(self) -> None:
        '''_compile_nodes writes an index number to the node's index field required for matrix lookup.'''
        i = 0
        for node in self.nodes:
            if node._linked is not None:
                continue
            node.set_index(i)
            i += 1

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
    
    def port(self, number: int) -> Source:
        return self.ports[number]

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
        node_dict = {node.unique(): False for node in self.nodes}

        # check if nodes are used in components
        for component in self.components:
            for node in component.nodes:
                node_dict[node.unique()] = True
        
        # check if nodes are used in terminals
        for source in self.sources:
            node_dict[source.ground_node] = True
            node_dict[source.node] = True
            node_dict[source.source_node] = True
        
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
        Runs an S-parameter analysis using the MNA method for the network at the specified frequencies.

        Parameters:
        -----------
        frequencies (np.ndarray): An array of frequencies at which to run the analysis.

        Returns:
        --------
        Sparameters: An Sparameters object containing the S-parameter matrix for the network at the specified frequencies

        """
        self._compile_nodes()
        self._check_unconnected_nodes()
        
        M = len(self.sources)
        nF = len(frequencies)
        N = max([node.index for node in self.nodes]) + 1


        component_compilers = [c._generate_compiler_function() for c in self.components]
        
        ntot = M * nF

        G = np.zeros((N, N, nF), dtype=np.complex128)
        B = np.zeros((N, M, nF), dtype=np.complex128)
        D = np.zeros((M, M, nF), dtype=np.complex128)
        A = np.zeros((M+N, M+N, nF), dtype=np.complex128)

        for compiler in component_compilers:
            G = compiler(G, frequencies)

        for i, source in enumerate(self.sources):
            B[source.ground.index,i] = -1.0
            B[source.source_node.index,i] = 1.0
        
        D = np.transpose(B,(1,0,2))

        voltage_source_nodes = [(t.node.index, t.source_node.index, t.ground_node.index) for t in self.sources]

        A[:N,:N,:] = G 
        A[:N, N:,:] = B
        A[N:,:N,:] = D

        Zs = np.zeros((M,nF), dtype=np.complex128)
        for it in range(M):
            Zs[it,:] = self.sources[it].z_source(frequencies)


        indices = np.array(voltage_source_nodes).astype(np.int32)
        frequencies = np.array(frequencies).astype(np.float32)
        Sol = None

        if self.suppress_loadbar:
            V, Sol = solve_MNA_RF_nopgb(A, Zs, indices, frequencies)
        else:
            with nbp.ProgressBar(total=ntot) as progress:
                V, Sol = solve_MNA_RF(A, Zs, indices, frequencies, progress) 
        
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
            [node_from, node_to], functionlist, component_value=current
        ).set_metadata(**Components.CURRENTSOURCE)
        self.sources.append(current_source_obj)
        return current_source_obj

    def terminal(self, signal_node: Node, Z0: float | Scalar, gnd_node: Node = None) -> Source:
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
        
        source_node = self.named_node('IntermediatePortNode')
        Z0 = parse_numeric(Z0)
        impedance_component = self.impedance(source_node, signal_node, Z0, display_value=Z0).set_metadata(**Components.RESISTOR)
        terminal_object = Source(gnd_node, signal_node, source_node, impedance_component)
        self.sources.append(terminal_object)
        self.ports[len(self.sources)] = terminal_object
        return terminal_object

    def new_port(self, impedance: float) -> Node:
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

        admittance_component = Component([node1, node2], functionlist, component_value=display_value).set_metadata(**Components.ADMITTANCE)
        admittance_component._impedance = admittance_simvalue.inverse()
        self.components.append(admittance_component)
        return admittance_component

    def impedance(self, node1: Node, node2: Node, Z: float,
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
        impedance_object = Component([node1, node2], functionlist, component_value=display_value).set_metadata(**Components.IMPEDANCE)
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
        
        return self.impedance(node1, node2, R, display_value=R).set_metadata(**Components.RESISTOR)

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
        
        return self.admittance(node1, node2, admittance, display_value=C).set_metadata(**Components.CAPACITOR)
        
        
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
        
        return self.admittance(node1, node2, admittance, display_value=L).set_metadata(**Components.INDUCTOR)
    
        
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
            [gnd, port1, port2], functionlist, component_value=Z0
        ).set_metadata(**Components.TRANSMISSIONLINE)
        self.components.append(transmissionline_component)
        return transmissionline_component

    def TL(self, node1: Node, node2: Node, beta: float | Scalar, length: float | Scalar, Z0: float | Scalar) -> Component:
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
        
        comp = self.n_port_Y(self.gnd, [node1, node2], [[y11, y12], [y21, y22]], Z0).set_metadata(value = Z0, **Components.TRANSMISSIONLINE)
        return comp
    
    def n_port_S(
            self,
            gnd: Node,
            nodes: list[Node],
            Sparam: list[list[Callable]],
            Z0: float,
    ) -> Component:
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
        component = Component(nodes + [gnd, ],[ComponentFunction(nodes + [gnd, ],Function(comp_function),True),], Z0 ).set_metadata(value=Z0, **Components.NPORT)
        self.components.append(component)
        return component

    def n_port_Y(
            self,
            gnd: Node,
            nodes: list[Node],
            Yparams: list[list[Callable]],
            Z0: float,
    ) -> Component:
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
        component = Component(nodes + [gnd, ],[ComponentFunction(nodes + [gnd, ],Function(comp_function),True),], Z0 ).set_metadata(value=Z0, **Components.NPORT)
        self.components.append(component)
        return component
    
    

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

    # def two_port_reciprocal(
    #     self,
    #     gnd: Node,
    #     port1: Node,
    #     port2: Node,
    #     S11: complex,
    #     S12: complex,
    #     S22: complex,
    #     Z0: float,
    # ) -> tuple[Component, Component, Component]:
    #     """
    #     Calculate the admittance parameters for a two-port reciprocal network.
    #     Args:
    #         gnd (Node): The ground node.
    #         port1 (Node): The first port node.
    #         port2 (Node): The second port node.
    #         S11 (complex): The S11 scattering parameter as a function of frequency.
    #         S12 (complex): The S12 scattering parameter as a function of frequency.
    #         S22 (complex): The S22 scattering parameter as a function of frequency.
    #         Z0 (float): The characteristic impedance.
    #     Returns:
    #         tuple[Component, Component, Component]: A tuple containing the admittance components Y1, Y2, and Y3.
    #     """
    #     Z0 = parse_numeric(Z0)
    #     def detS(f):
    #         return ((1 + S11(f)) * (1 + S22(f))) - S12(f) ** 2
    #     def Y11(f):
    #         return ((1 - S11(f)) * (1 + S22(f)) + S12(f) ** 2) / (detS(f)) * 1 / Z0(f)
    #     def Y12(f):
    #         return -2 * S12(f) / detS(f) * 1 / Z0(f)
    #     def Y22(f):
    #         return ((1 + S11(f)) * (1 - S22(f)) + S12(f) ** 2) / (detS(f)) * 1 / Z0(f)

    #     Y1 = self.admittance(gnd, port1, lambda f: Y11(f) + Y12(f))
    #     Y2 = self.admittance(port1, port2, lambda f: -Y12(f))
    #     Y3 = self.admittance(gnd, port2, lambda f: Y22(f) + Y12(f))

    #     return (Y1, Y2, Y3)