from __future__ import annotations
from enum import Enum
from typing import List, Callable, Literal
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from .solver import solve_MNA_RF, solve_MNA_RF_nopgb, solve_MNA_DC
import numba_progress as nbp
from loguru import logger

from .sparam import Sparameters
from .numeric import SimParam, enforce_simparam, Function, SimParam, Scalar, ensure_simparam

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

def _stack(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    ''' Copies a 2D matrix along the 3rd using the size of the provided vector.
     
    >>> _stack(np.array([[1,2],[3,4]]), np.array([1,2,3])).shape
    (2, 2, 3)
      
    '''
    if vector is None or isinstance(vector, (int, float)):
        return matrix

    out = np.tile(np.expand_dims(matrix, axis=2), (1,1,vector.shape[0]))

    return out

def _get_power(number: float):
    ''' Splits a float into a base value and the power of 10 in groups of 1000.
    
    >>> _get_power(4532)
    (4.532, 3)
    '''
    N_thousands = np.floor(np.log10(np.abs(number)) / 3) * 3
    N_thousands = min(12, max(-12, N_thousands))
    base_number = number / (10**N_thousands)
    return base_number, N_thousands

def format_value_units(value: float, unit: str) -> str:
    """ Formats a value with units for display. 
    
    >>> format_value_units(4532, 'Hz')
    '4.53 kHz'
    """
    
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

_TWO_MAT = np.array([[1,-1],[-1,1]])

class Components:
    RESISTOR = {'name': 'Resistor', 'unit': 'Ω'}
    CAPACITOR = {'name': 'Capacitor', 'unit': 'F'}
    INDUCTOR = {'name': 'Inductor', 'unit': 'H'}
    CURRENTSOURCE = {'name': 'Current Source', 'unit': 'A'}
    VOLTAGESOURCE = {'name': 'Voltage Source', 'unit': 'V'}
    IMPEDANCE = {'name': 'Impedance', 'unit': 'Ω'}
    ADMITTANCE = {'name': 'Admittance', 'unit': 'G'}
    TRANSMISSIONLINE = {'name': 'Transmission Line', 'unit': 'Ω'}
    NPORT = {'name': 'N-Port', 'unit': 'Ω'}
    CUSTOM = {'name': 'Custom', 'unit': 'None'}

##### CONSTANTS

TWOPI = np.float64(2 * np.pi)
PI = np.pi
GEN_NULL = Scalar(0)
NPZERO = np.float64(0)

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
            return f"{self.name}[{self._index}]"
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
            self._linked = other
            return other
        return NotImplemented



@dataclass
class ComponentFunction:
    """ ComponentFunction class for the Component object. """
    idx_generators: list[list[Node]]
    function: SimParam
    
    @property
    def _slice(self):
        return self.gen_slice()
    
    def gen_slice(self):
        if len(self.idx_generators) == 1:
            return np.ix_([n.index for n in self.idx_generators[0]])
        else:
            return np.ix_(*[[n.index for n in nodes] for nodes in self.idx_generators])
        

class Compilable:

    is_linear: bool = False

    def _get_functions(self, type: Literal['G','B','C','D','I']) -> list[ComponentFunction]:
        return [f for f in self.functions if f.matrix == type]

    def _generate_compiler(self, simulation: Literal['SP','DC','AC']) -> Callable:
        '''Generates a callable that will plug in the components matrix entries for a given frequency.'''


        if simulation == 'SP':
            def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
                for function in self.functions:
                    matrix[function.gen_slice()] += function.function(f)
                return matrix
        elif simulation == 'DC':
            def compiler(matrix: np.ndarray, v: np.ndarray) -> np.ndarray:
                for function in self.functions:
                    matrix[function.gen_slice()] += function.function(NPZERO)
                return matrix
        elif simulation == 'AC':
            def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
                for function in self.functions:
                    matrix[function.gen_slice()] += function.function(f)
                return matrix
        return compiler

   
    def _generate_I_compiler(self, simulation: Literal['SP','DC','AC']) -> Callable:
        '''Generates a callable that will plug in the components matrix entries for a given frequency.'''
        if simulation == 'SP':
            def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
                for function in self.source_functions:
                    matrix[function.gen_slice()] += function.function(f)
                return matrix
        elif simulation == 'DC':
            def compiler(matrix: np.ndarray, v: np.ndarray) -> np.ndarray:
                for function in self.source_functions:
                    matrix[function.gen_slice()] += function.function(NPZERO).squeeze()
                return matrix
        elif simulation == 'AC':
            def compiler(matrix: np.ndarray, f: float) -> np.ndarray:
                for function in self.source_functions:
                    matrix[function.gen_slice()] += function.function(f)
                return matrix
        return compiler
    

class Component(Compilable):
    """ Component class for the Network object. 
    This class represents a component in the network, such as a resistor, capacitor, inductor, etc.
    
    Parameters
    ----------
    nodes : list[Node]
        A list of Node objects corresponding to the nodes the component is connected to.
    functions : list[ComponentFunction]
        A list of ComponentFunction objects corresponding to the functions of the component.
    source_functions : list[ComponentFunction]
        A list of ComponentFunction objects corresponding to the source functions of the component.
    component_value : SimParam
        The value of the component.
    non_linear : bool
        A boolean indicating if the component is non-linear.

    """
    is_linear: bool = True
    def __init__(
        self, nodes: list[Node], 
        functions: list[ComponentFunction] = None, 
        source_functions: list[ComponentFunction] = None,
        component_value: SimParam = None,
    ):
        self.nodes: list[Node] = nodes
        self.functions: list[ComponentFunction] = functions
        self.source_functions: list[ComponentFunction] = source_functions
        self._component_value: SimParam = enforce_simparam(component_value)
        self._impedance: SimParam = None
        self.meta_data: dict[str, str] = dict()

        if self.source_functions is None:
            self.source_functions = []

        if self.functions is None:
            self.functions = []

        self.component_name: str = None
        self.component_unit: str = None

    def print_data(self) -> None:
        '''Prints the component data.'''
        print('')
        print(f'Component: {self.component_name}')
        print(f'Nodes: {[str(n) for n in self.nodes]}')
        print(f'Value: {self.display_value}')
        for key, value in self.meta_data.items():
            print(f'{key}: {format_value_units(value.scalar(), value.unit)}')
        print('')

    @property
    def display_value(self) -> str:
        '''Returns the value of the component with units.'''
        return format_value_units(self._component_value.scalar(), self.component_unit)
    
    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        value = self.display_value
        return f"{self.component_name}: {[str(n) for n in self.nodes]}, value={value}"
    
    def set_metadata(self, name: str = 'Component', unit: str = '', value: float | SimParam = None, **kwargs: dict[str, str]) -> Component:
        '''Sets the metadata for the component.'''
        self.component_name = name
        self.component_unit = unit
        if value is not None:
            self._component_value = enforce_simparam(value)
        for key, value in kwargs.items():
            self.meta_data[key] = enforce_simparam(value)
        return self
    
class LinearComponent(Component):
    is_linear: bool = True

class NonLinearComponent(Component):
    is_linear: bool = False

    def __init__(
        self, nodes: list[Node], 
        functions: list[ComponentFunction] = None, 
        source_functions: list[ComponentFunction] = None,
        component_value: SimParam = None,
        v_function: Callable[[np.ndarray], np.ndarray] = None,
    ):
        super().__init__(nodes, functions, source_functions, component_value)
        self.v_function = v_function
        
    
@dataclass
class Source(Compilable):
    ground: Node
    source_node: Node
    output_node: Node = None
    source_impedance: Component = None
    small_signal: bool = False
    dc_voltage: float = None
    ac_voltage: float = None
    index: int = None
    functions: list[ComponentFunction] = None, 
    source_functions: list[ComponentFunction] = None

    def __post_init__(self):
        if self.source_functions is None:
            self.source_functions = []
        if self.functions is None:
            self.functions = []
            
    @property
    def in_sparam(self) -> bool:
        return self.small_signal

    @property
    def in_dc(self) -> bool:
        return isinstance(self.dc_voltage, (int, float, SimParam))
    
    @property
    def in_ac(self) -> bool:
        return isinstance(self.ac_voltage, (int, float, SimParam))
    
    @property
    def ground_node(self) -> Node:
        return self.ground
    
    @property
    def z_source(self) -> SimParam:
        return self.source_impedance._impedance

    def print_data(self):
        print(f'Ground: {self.ground}')
        print(f'Node: {self.output_node}')
        print(f'Source Node: {self.source_node}')
        print(f'Source Impedance: {self.source_impedance}')
        print(f'DC Voltage: {self.dc_voltage}')
        print(f'AC Voltage: {self.ac_voltage}')

    
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
            comp.print_data()

    @property
    def node_names(self) -> list[str]:
        '''A list of strings corresponding to each node.'''
        return [n.name for n in self.nodes]

    @property
    def dc_sources(self) -> list[Source]:
        '''Returns a list of sources that are DC sources.'''
        return [source for source in self.sources if source.in_dc]
    
    @property
    def ac_sources(self) -> list[Source]:
        '''Returns a list of sources that are AC sources.'''
        return [source for source in self.sources if source.in_ac]
    
    @property
    def sparam_sources(self) -> list[Source]:
        '''Returns a list of sources that are S-parameter sources.'''
        return [source for source in self.sources if source.in_sparam]
    
    def unlinked_nodes(self) -> list[Node]:
        '''Returns a list of nodes that are not linked to any other nodes.'''
        return [node for node in self.nodes if node._linked is None]
    
    def _define_indices(self) -> None:
        '''_define_indices writes an index number to the node's index field required for matrix lookup.
        
        This method is called before running the analysis to ensure that all nodes have an index number.
        '''
        i = 0
        for node in self.nodes:
            if node._linked is not None:
                continue
            node.set_index(i)
            i += 1
        
        for source in self.sources:
            source.index = i
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
            node_dict[source.output_node] = True
            node_dict[source.source_node] = True
        
        for node, used in node_dict.items():
            if not used:
                logger.error(f"Node {node.name} is not connected to any components.")
                logger.error("Unconnected nodes will cause the analysis to yield 0 values.")
                raise ValueError(f"Node {node.name} is not connected to any components.")

    def _has_nonlinear_components(self) -> bool:
        '''Returns True if the network contains any non-linear components.'''
        return any([component.is_linear for component in self.components])
    
    def run_dc(self, initial_voltage: float = 0, maxiter: int = 5) -> np.ndarray:

        self._define_indices()
        self._check_unconnected_nodes()

        if not self._has_nonlinear_components():
            maxiter = 1

        non_linear_components: list[NonLinearComponent] = [c for c in self.components if not c.is_linear]
        linear_components: list[LinearComponent] = [c for c in self.components if c.is_linear]
        source_components: list[Source] = [s for s in self.sources]

        M = len(self.sources)
        N = max([node.index for node in self.nodes]) + 1

        Vsol = np.zeros((N, ), dtype=np.complex128) + np.linspace(0, initial_voltage, N)
        Vsol2 = np.zeros((N + M, ), dtype=np.complex128)
        Vsol2[:N] = np.linspace(0, initial_voltage, N)

        A_compilers = [c._generate_compiler('DC') for c in self.components + self.sources]
        I_compilers = [c._generate_I_compiler('DC') for c in self.components + self.sources]

        Jnl_compilers = [c._generate_compiler('DC') for c in non_linear_components]
        Jl_compilers = [c._generate_compiler('DC') for c in linear_components]
        source_compilers = [c._generate_compiler('DC') for c in source_components]

        A = np.zeros((M+N, M+N), dtype=np.complex128)

        Jnl = np.zeros((M+N, M+N), dtype=np.complex128)
        Jl = np.zeros((M+N, M+N), dtype=np.complex128)

        SolVec = np.zeros((M+N,), dtype=np.complex128)
        SolVec2 = np.zeros((M+N,), dtype=np.complex128)
        for i in range(maxiter):

            for component in non_linear_components:
                component.v_function(Vsol)

            for compiler in A_compilers:
                A = compiler(A, Vsol)

            
            for compiler in I_compilers:
                SolVec = compiler(SolVec, Vsol)
                

            ## Method One
            Vsol = solve_MNA_DC(A, SolVec, N, M) 
            print('Vout=',Vsol.real)
            print('')
            ## Method Two

            for component in non_linear_components:
                component.v_function(Vsol2[:N])

            for compiler in Jnl_compilers:
                Jnl = compiler(Jnl, Vsol2[:N])

            for compiler in Jl_compilers + source_compilers + Jnl_compilers:
                Jl = compiler(Jl, Vsol2[:N])
            
            for compiler in I_compilers:
                SolVec2 = compiler(SolVec2, Vsol2[:N])

            
            Vsol2[1:] = np.linalg.pinv(Jl[1:,1:]) @ (SolVec2[1:] - Jnl[1:,1:] @ Vsol2[1:])

            print('Vou 2 =',Vsol2.real)

            input('')
        return Vsol
        
        
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
        self._define_indices()
        self._check_unconnected_nodes()
        
        M = len(self.sources)
        nF = len(frequencies)
        N = max([node.index for node in self.nodes]) + 1

        A_compilers = [c._generate_compiler('SP') for c in self.components + self.sources]
        
        ntot = M * nF

        A = np.zeros((M+N, M+N, nF), dtype=np.complex128)

        for compiler in A_compilers:
            A = compiler(A, frequencies)

        Zs = np.zeros((M,nF), dtype=np.complex128)

        voltage_source_nodes = []
        for isource, source in enumerate(self.sources):
            if not source.in_sparam:
                continue
            voltage_source_nodes.append((isource, source.output_node.index, source.source_node.index, source.ground_node.index))
            Zs[isource,:] = source.z_source(frequencies)

        indices = np.array(voltage_source_nodes).astype(np.int32)
        frequencies = np.array(frequencies).astype(np.float32)
        Sol = None

        if self.suppress_loadbar:
            V, Sol = solve_MNA_RF_nopgb(A, Zs, indices, frequencies, N, M)
        else:
            with nbp.ProgressBar(total=ntot) as progress:
                V, Sol = solve_MNA_RF(A, Zs, indices, frequencies, N, M, progress) 
        
        return Sparameters(Sol, frequencies)

    def terminal(self, signal_node: Node, Z0: float | SimParam, gnd_node: Node = None) -> Source:
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
        Z0 = enforce_simparam(Z0, unit='Ω')
        impedance_component = self.impedance(source_node, signal_node, Z0, display_value=Z0).set_metadata(**Components.RESISTOR)
        
        
        terminal_object = Source(gnd_node,source_node,output_node=signal_node,
                                 source_impedance=impedance_component, 
                                 small_signal=True)
        
        Bcf = ComponentFunction([[gnd_node, source_node],[terminal_object]], lambda f: _stack(np.array([[-1],[1]]),f))
        Ccf = ComponentFunction([[terminal_object],[gnd_node, source_node]], lambda f: _stack(np.array([[1,-1]]),f))
        
        terminal_object.functions = [Bcf, Ccf]
        terminal_object.source_functions = []    
        self.sources.append(terminal_object)
        self.ports[len(self.sources)] = terminal_object
        return terminal_object

    def DC_source(self, node: Node, voltage: float | SimParam, gnd: Node = None) -> Source:
        
        if gnd is None:
            gnd = self.gnd
        
        voltage = enforce_simparam(voltage, unit='V')

        terminal_object = Source(ground=gnd,
                                 source_node=node, dc_voltage=voltage)
        
        Bcf = ComponentFunction([[gnd, node],[terminal_object]], lambda f: _stack(np.array([[-1],[1]]),f))
        Ccf = ComponentFunction([[terminal_object],[gnd, node]], lambda f: _stack(np.array([[-1, 1]]),f))
        Dcf = ComponentFunction([[terminal_object],[terminal_object]], lambda f: _stack(np.array([[0]]),f))
        Icf = ComponentFunction([[terminal_object]], lambda f: _stack(np.array([voltage.value]),f))

        terminal_object.functions = [Bcf, Ccf, Dcf]
        terminal_object.source_functions = [Icf]
        self.sources.append(terminal_object)
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
        
        admittance_simvalue = enforce_simparam(Y, unit='S')

        if display_value is None:
            display_value = admittance_simvalue
            logger.debug(f'Defaulting display value to {display_value}')

        
        def _G_stencil(f: np.ndarray) -> np.ndarray:
            return admittance_simvalue(f) * _stack(_TWO_MAT, f)
        
        functionlist = [ComponentFunction([[node1, node2],[node1, node2]], _G_stencil),]
        admittance_component = Component([node1, node2], functions=functionlist, component_value=display_value).set_metadata(**Components.ADMITTANCE)
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
        
        impedance = enforce_simparam(Z, unit='Ω')
        
        if display_value is None:
            display_value = impedance(1)
            logger.debug(f'Defaulting display value to {display_value}')
        
        def _G_stencil(f: float) -> np.ndarray:
            return (1/impedance(f)) * _stack(_TWO_MAT, f)
        
        functionlist = [ComponentFunction([[node1, node2],[node1, node2]], _G_stencil),]

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
        C = enforce_simparam(C, unit='F')
        
        def _Y_function(f):
            return 1j * TWOPI * f * C(f)
        
        admittance = Function(_Y_function)
        
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
        L = enforce_simparam(L, unit='H')
        
        def admittance_f(f) -> np.ndarray:
            return 1/(1j * TWOPI * f * L(f))
        
        admittance = Function(admittance_f, inf=1e9)
        
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
        func_er = enforce_simparam(er)
        func_Z0 = enforce_simparam(Z0, unit='Ω')
        func_L = enforce_simparam(L, unit='m')

        def beta(f):
            return TWOPI * f / c0 * np.sqrt(func_er(f))
        
        return self.TL(port1, port2, Function(beta), func_L, func_Z0)

    def TL(self, node1: Node, node2: Node, beta: float | SimParam, length: float | SimParam, Z0: float | SimParam) -> Component:
        beta = enforce_simparam(beta, unit='rad/m')
        length = enforce_simparam(length, unit='m')
        Z0 = enforce_simparam(Z0, unit='Ω')

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
        component = Component(nodes + [gnd, ],[ComponentFunction(comp_function),], Z0).set_metadata(value=Z0, **Components.NPORT)
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
            Y = np.array([[y(f) for y in row] for row in Yparams], dtype=np.complex128)
            Y2 = np.zeros((N+1,N+1,nF),dtype=np.complex128)
            Y2[:N,:N,:] = Y
            for i in range(N):
                Y2[i,N,:] = -np.sum(Y[i,:,:],axis=0)
                Y2[N,i,:] = -np.sum(Y[:,i,:],axis=0)
                Y2[N,N,:] += np.sum(Y[i,:,:],axis=0)
            return Y2
        component = Component(nodes + [gnd, ], functions=[ComponentFunction([nodes, nodes], comp_function)], component_value=Z0).set_metadata(value=Z0, **Components.NPORT)
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