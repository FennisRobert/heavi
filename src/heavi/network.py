########################################################################################
##
##    The Network Class
##    The Heavi Network class manages the components and nodes in the circuit. 
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################

#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------


from __future__ import annotations
from enum import Enum
from typing import List, Callable, Literal
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from .solvers import solve_MNA_RF, solve_MNA_RF_nopgb, solve_MNA_DC, solve_MNA_TRAN, solve_MNA_TRAN_nopgb
from .node import Node, NodeAllocator
from .sparam import Sparameters
from .numeric import SimParam, enforce_simparam, Function, SimParam, Scalar
from .base import (MatSource, BaseComponent, SimulationType, Port, Admittance, Resistor, Inductor, 
                   Capacitor, Impedance, TransmissionLine, NPortS, NPortY, VoltageDC, Diode, VoltageFN)

from loguru import logger
import numba_progress as nbp

#  __   __        __  ___           ___  __  
# /  ` /  \ |\ | /__`  |   /\  |\ |  |  /__` 
# \__, \__/ | \| .__/  |  /~~\ | \|  |  .__/ 
# -------------------------------------------


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


TWOPI = np.float64(2 * np.pi)
PI = np.pi
GEN_NULL = Scalar(0)
NPZERO = np.float64(0)

#  __             __   __  
# /  ` |     /\  /__` /__` 
# \__, |___ /~~\ .__/ .__/ 
# -------------------------------------------

    
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
        
        self.components: list[BaseComponent] = []
        self.sources: list[MatSource] = []
        self.ports: dict[int, Port] = {}
        self.port_counter: int = 1
        
        self.suppress_loadbar: bool = suppress_loadbar
        self._vramp: float = 0

        self._default_node_index: str = default_name
        self._nodes: list[Node] = [self.gnd]
        self._node_library: dict[str, NodeAllocator] = {self._default_node_index: NodeAllocator(self._default_node_index, self)}
        
        self._initialized: bool = False

    def get_vramp(self):
        return self._vramp
    
    def print_components(self) -> None:
        '''Prints an overview of the components in the Network'''
        for comp in self.components:
            print(comp.summary())

    @property
    def node_names(self) -> list[str]:
        '''A list of strings corresponding to each node.'''
        return [n.name for n in self._nodes]
    
    def unlinked_nodes(self) -> list[Node]:
        '''Returns a list of nodes that are not linked to any other nodes.'''
        return [node for node in self._nodes if node._linked is None]
    
    def _define_indices(self, simtype: SimulationType) -> None:
        '''_define_indices writes an index number to the node's index field required for matrix lookup.
        
        This method is called before running the analysis to ensure that all nodes have an index number.
        '''

        i = 0
        for node in self._nodes:
            if node._linked is not None:
                continue
            node.set_index(i)
            i += 1
        
        for component in self.components:
            N = component.n_requested_sources(simtype)
            sources = [MatSource(i) for i in range(i, i+N)]
            self.sources.extend(sources)
            component.set_sources(sources)
            i += N

    
    def _assign_internal_nodes(self, simtype: SimulationType) -> None:
        '''Assigns internal nodes to components that require them.'''
        for component in self.components:
            N = component.n_requested_internal_nodes(simtype)
            nodes = [self.node(index='core_internal') for _ in range(N)]
            component.set_internal_nodes(nodes)
    
    def port(self, number: int) -> BaseComponent | None:
        """Returns the port object corresponding to the provided number.

        Args:
            number (int): The port number to retrieve.

        Returns:
            BaseComponent | None: The port object corresponding to the provided number.
        """        
        return self.ports.get(number, None)

    def get_port_index(self) -> int:
        """Returns the next available port index.

        Returns:
            int: The next available port index.
        """        
        index = self.port_counter
        self.port_counter += 1
        return index
    
    def node(self, number: int = None, index: str = None) -> Node:
        """Return the node with the provided number from the provided node index library.

        Args:
            number (int, optional): The noden umber. Defaults to None.
            index (str, optional): The index. Defaults to None. (default index)

        Returns:
            Node: The node object
        """        
        if index is None:
            index = self._default_node_index
        if index not in self._node_library:
            self._node_library[index] = NodeAllocator(index, self)
        
        node, is_new = self._node_library[index].request(number)
        if is_new:
            self._nodes.append(node)
        return node


    def mnodes(self, N: int, name: str = None) -> list[Node]:
        """Creates a list of N nodes in the network.

        Args:
            N (int): The number of nodes to create.
            name (str, optional): The node name prefix. Defaults to None.

        Returns:
            list[Node]: A list of N nodes.
        """        
        if name is None:
            return [self.node() for _ in range(N)]
        else:
            return [self.node(f'{name}_{i+1}') for i in range(N)]
    
    def _check_unconnected_nodes(self) -> None:
        '''Checks for unconnected nodes in the network and raises a warning if any are found.'''
        # collecct a list of nodes and included status
        node_dict = {node.unique(): False for node in self._nodes}

        # check if nodes are used in components
        for component in self.components:
            for node in component.all_nodes:
                node_dict[node.unique()] = True
        
        for node, used in node_dict.items():
            if not used:
                logger.error(f"Node {node.name} is not connected to any components.")
                logger.error("Unconnected nodes will cause the analysis to yield 0 values.")
                raise ValueError(f"Node {node.name} is not connected to any components.")

    def _get_components(self, simtype: SimulationType) -> list[BaseComponent]:
        '''Returns a list of components that are compatible with the simulation type.'''
        component_list = []
        for component in self.components:
            if simtype in component.supported_simulations:
                component_list.append(component)
            else:
                logger.warning(f"Component {component.component_name} does not support {simtype} simulation. Component is ignored")
        return component_list
    
    def run_dc(self, 
               maxiter: int = 100, 
               source_ramp_steps: int = 0,
               voltage_interpolation_steps: int = 201) -> np.ndarray:
        """Runs a DC analysis on the network.

        Args:
            maxiter (int, optional): _description_. Defaults to 100.
            source_ramp_steps (int, optional): _description_. Defaults to 0.
            voltage_interpolation_steps (int, optional): _description_. Defaults to 201.

        Returns:
            np.ndarray: _description_
        """        
        logger.warning("DC Analysis is still under development and will not work for most components.")

        simtype = SimulationType.DC

        self._assign_internal_nodes(simtype)
        self._define_indices(simtype)
        self._check_unconnected_nodes()
        if all([c.is_linear(simtype) for c in self._get_components(simtype)]):
            maxiter = 1
            
        M = len(self.sources)
        N = max([node.index for node in self._nodes]) + 1

        Vsol = np.zeros((N, ))
        
        A_compilers = [c.DC_matrix_compiler() for c in self.components]
        I_compilers = [c.DC_source_compiler() for c in self.components]
        A_compilers = [c for c in A_compilers if c is not None]
        I_compilers = [c for c in I_compilers if c is not None]

        SolVec = np.zeros((M+N,))
        
        if source_ramp_steps == 0:
            rampvals = np.array([1.0,])
        else:
            rampvals = np.linspace(0, 1, source_ramp_steps+2)[1:]


        A = np.zeros((M+N, M+N)) 

        DiodeData = [(-1,-1,np.linspace(0,1,5),np.linspace(0,1,5)),]
        ## 
        for comp in self.components:
            if not isinstance(comp, Diode):
                continue
            I, V = comp.IVData(Nsteps=voltage_interpolation_steps)
            DiodeData.append((comp.nodes[0].index, comp.nodes[1].index, I,V))

        for A_compiler in A_compilers:
            A = A_compiler(A)
        
        for I_compiler in I_compilers:
            SolVec = I_compiler(SolVec)

        Vsol = solve_MNA_DC(A, SolVec, N, M, rampvals, maxiter, DiodeData, 0.001, 0.001) 

        return Vsol
    
    def run_transient(self, duration: float, dt: float, alpha: float = 0.5,
                      maxiter: int = 5, 
                      source_ramp_steps: int = 0,
                      voltage_interpolation_steps: int = 51) -> np.ndarray:
        
        logger.warning("Transient Analysis is still under development and will not work for most components.")

        simtype = SimulationType.TRANS

        self._assign_internal_nodes(simtype)
        self._define_indices(simtype)
        self._check_unconnected_nodes()

        if all([c.is_linear(simtype) for c in self._get_components(simtype)]):
            maxiter = 1
        
        timesteps = np.linspace(0, duration, int(duration/dt)+1)
        n_timesteps = len(timesteps)

        M = len(self.sources)
        N = max([node.index for node in self._nodes]) + 1

        Vsol = np.zeros((N, ))
        
        A_compilers = [c.TRANS_matrix_compiler() for c in self.components]
        I_compilers = [c.TRANS_source_compiler() for c in self.components]
        A_compilers = [c for c in A_compilers if c is not None]
        I_compilers = [c for c in I_compilers if c is not None]

        SolVec = np.zeros((M+N,))
        
        if source_ramp_steps == 0:
            rampvals = np.array([1.0,])
        else:
            rampvals = np.linspace(0, 1, source_ramp_steps+2)[1:]

        A = np.zeros((M+N, M+N)) 

        DiodeData = [(-1,-1,np.linspace(0,1,5),np.linspace(0,1,5)),]
        LCData = [(-1, -1,-1,-1, 0.0),]

        ## GENERATE TIMESERIES
        transient_source_data = [(-1,np.linspace(0,1,5),np.linspace(0,1,5)),]
        for source in self.components:
            if isinstance(source, VoltageFN):
                transient_source_data.append((source.index, source.vfun(timesteps), timesteps))
        
        for comp in self.components:
            if isinstance(comp, Diode):
                I, V = comp.IVData(Nsteps=voltage_interpolation_steps)
                DiodeData.append((comp.nodes[0].index, comp.nodes[1].index, I,V))
            elif isinstance(comp, Capacitor):
                LCData.append((0, comp.nodes[0].index, comp.nodes[1].index, -1, comp.C))
            elif isinstance(comp, Inductor):
                LCData.append((1, comp.nodes[0].index, comp.nodes[1].index, comp.sources[0].index, comp.L))

        for A_compiler in A_compilers:
            A = A_compiler(A)
        
        for I_compiler in I_compilers:
            SolVec = I_compiler(SolVec)

        if self.suppress_loadbar:
            Vsol = solve_MNA_TRAN_nopgb(A, SolVec, N, M, transient_source_data, rampvals, maxiter, DiodeData) 

        else:
            with nbp.ProgressBar(total=n_timesteps) as progress:
                Vsol = solve_MNA_TRAN(A, SolVec, N, M, transient_source_data, rampvals, maxiter, DiodeData, LCData, alpha, progress) 

        
        return timesteps, Vsol
        
    def run_sp(self, frequencies: np.ndarray, reinitialize: bool = False) -> Sparameters:
        """
        Runs an S-parameter analysis using the MNA method for the network at the specified frequencies.

        Parameters:
        -----------
        frequencies (np.ndarray): An array of frequencies at which to run the analysis.

        Returns:
        --------
        Sparameters: An Sparameters object containing the S-parameter matrix for the network at the specified frequencies

        """

        # Prepare the network for simulation

        simtype = SimulationType.SP
        
        if not self._initialized:
            self._assign_internal_nodes(simtype)
            self._define_indices(simtype)
            self._check_unconnected_nodes()
            self._initialized = True

        # Define the number of sources and nodes

        M = len(self.sources)
        nF = len(frequencies)
        N = max([node.index for node in self._nodes]) + 1
        
        # Initialize the S-parameter matrix and voltage source compiler
        A_compilers = [c.SP_matrix_compiler() for c in self._get_components(simtype)]
        A_compilers = [c for c in A_compilers if c is not None]


        total_number_of_computations = M * nF

        # Initialize the S-parameter matrix and voltage source compiler
        mna_matrix = np.zeros((M+N, M+N, nF), dtype=np.complex128)
        for compiler in A_compilers:
            mna_matrix = compiler(mna_matrix, frequencies)

        # Initialize the vector containing all impedances
        impedance_vector = np.zeros((M,nF), dtype=np.complex128)

        # Initialize the vector containing all S-parameter port indices.
        voltage_source_nodes = []
        for iport, port in self.ports.items():
            voltage_source_nodes.append((iport-1, port.sig_node.index, port.int_node.index, port.gnd_node.index))
            impedance_vector[iport-1,:] = port.Z0(frequencies)
        indices = np.array(voltage_source_nodes).astype(np.int32)

        # Initialize the frequencies array in the right data format
        frequencies = np.array(frequencies).astype(np.float32)

        Sol = None

        # Run the simulation
        if self.suppress_loadbar:
            V, Sol = solve_MNA_RF_nopgb(mna_matrix, impedance_vector, indices, frequencies, N, M)
        else:
            with nbp.ProgressBar(total=total_number_of_computations) as progress:
                V, Sol = solve_MNA_RF(mna_matrix, impedance_vector, indices, frequencies, N, M, progress) 
        
        if reinitialize:
            self._initialized = False

        return Sparameters(Sol, frequencies)
    

    def DC_source(self, node: Node, voltage: float | SimParam, gnd: Node = None, r_series: float | SimParam = 0.001) -> VoltageDC:
        """Creates a DC voltage source between two nodes in the circuit.

        Args:
            node (Node): The node to which the voltage source is connected.
            voltage (float | SimParam): The voltage value of the source.
            gnd (Node, optional): The negative terminal node. Defaults to None.
            r_series (float | SimParam, optional): The series resistance. Defaults to 0.001.

        Returns:
            VoltageDC: The created DC voltage source component.
        """        
        if gnd is None:
            gnd = self.gnd
        vdc = VoltageDC(gnd, node, voltage, r_series=r_series)
        self.components.append(vdc)
        return vdc
    
    def TRANSIENT_source(self, node: Node,
                         voltage_function: Callable[[float], float],
                         gnd: Node = None,
                         r_series: float | SimParam = 0.001) -> VoltageFN:
        """Creates a transient voltage source between two nodes in the circuit.
        
        Args:
            node (Node): The node to which the voltage source is connected.
            voltage_function (Callable[[float], float]): The voltage function of the source.
            gnd (Node, optional): The negative terminal node. Defaults to None.
            r_series (float | SimParam, optional): The series resistance. Defaults to 0.001.
        
        Returns:
            VoltageFN: The created transient voltage source component.

        """
        if gnd is None:
            gnd = self.gnd
        vfn = VoltageFN(gnd, node, voltage_function, r_series=r_series)
        self.components.append(vfn)
        return vfn
    
    def new_port(self, Z0: float, node: Node = None, gnd: Node = None) -> Port:
        """Creates a new port plus its own output node in the network.

        Args:
            Z0 (float): The characteristic impedance of the port.
            node (Node, optional): The node to which the port is connected. Defaults to None.
            gnd (Node, optional): The reference ground node. Defaults to None.

        Returns:
            Port: The created port object.
        """        
        if gnd is None:
            gnd = self.gnd

        int_node = self.node()

        if node is None:
            src_node = self.node()
        else:
            src_node = node
        port = Port(self.get_port_index(), src_node, int_node, gnd, Z0)
        self.components.append(port)
        self.ports[port.index] = port
        
        return port
    
    def quick_port(self, Z0: float, gnd: Node = None) -> Node:
        """Creates a new port and returns only signal node of the port.

        Args:
            Z0 (float): The characteristic impedance of the port.
            gnd (Node, optional): The reference ground node. Defaults to None.

        Returns:
            Node: The signal node of the created port.
        """        
        port = self.new_port(Z0, gnd)
        return port.sig_node
    
    def admittance(self, node1: Node, node2: Node, Y: float) -> Admittance:
        """Creates and returns a component object for an admittance.

        Args:
            node1 (Node): The first node of the admittance.
            node2 (Node): The second node of the admittance.
            Y (float): The admittance value of the admittance in Siemens.

        Returns:
            Admittance: The created admittance component object.
        """        
        admittance_obj = Admittance(node1, node2, Y)
        self.components.append(admittance_obj)
        return admittance_obj

    def impedance(self, node1: Node, node2: Node, Z: float,
                  display_value: float = None) -> Impedance:
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
        Impedance: The created impedance component object.

        """
        imp = Impedance(node1, node2, Z)
        self.components.append(imp)
        return imp

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
        res = Resistor(node1, node2, R)
        self.components.append(res)
        return res
    
    def capacitor(self, node1: Node, node2: Node, C: float) -> Capacitor:
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
        cap = Capacitor(node1, node2, C)
        self.components.append(cap)
        return cap
        
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
        ind = Inductor(node1, node2, L)
        self.components.append(ind)
        return ind
    
    def diode(self, node1: Node, node2: Node,
              Is: float = 1.2e-15, n: float = 1, Vt: float = 0.0258) -> Diode:
        """
        Adds a diode component between two nodes in the circuit.
        Args:
            node1 (Node): The first node to which the diode is connected.
            node2 (Node): The second node to which the diode is connected.
            Is (float): The saturation current of the diode.
            n (float): The emission coefficient of the diode.
            Vt (float): The thermal voltage of the diode.
        Returns:
            Component: The created diode component.
        """
        diode = Diode(node1, node2, Vt, n, Is, 0.15)
        self.components.append(diode)
        return diode
    
    def transmissionline(
        self, port1: Node, port2: Node, Z0: float, er: float, L: float, gnd: Node = None
    ) -> TransmissionLine:
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
        c0 = 299792458
        func_er = enforce_simparam(er)
        func_Z0 = enforce_simparam(Z0, unit='Ω')
        def beta(f):
            return TWOPI * f / c0 * np.sqrt(func_er(f))
        
        return self.TL(port1, port2, Function(beta), L, func_Z0, gnd)

    def TL(self, node1: Node, node2: Node, beta: float | SimParam, length: float, Z0: float | SimParam, gnd: Node = None) -> TransmissionLine:
        if gnd is None:
            gnd = self.gnd
        tl = TransmissionLine(node1, node2, gnd, Z0, length, beta)
        self.components.append(tl)
        return tl
    
    def n_port_S(
            self,
            gnd: Node,
            nodes: list[Node],
            Sparam: list[list[Callable]],
            Z0: float,
    ) -> NPortS:
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
        nps = NPortS(nodes, gnd, Sparam, Z0)
        self.components.append(nps)
        return nps

    def n_port_Y(
            self,
            gnd: Node,
            nodes: list[Node],
            Yparams: list[list[Callable]],
            Z0: float,
    ) -> NPortY:
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
        npy = NPortY(nodes, gnd, Yparams, Z0)
        self.components.append(npy)
        return npy
