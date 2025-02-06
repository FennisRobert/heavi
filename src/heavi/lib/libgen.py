from __future__ import annotations

from ..rfcircuit import Network, Node


class BaseComponent:

    def __init__(self):
        self.network: Network = None
        self.nodes: dict[int, Node] = {}
        self.gnd: Node = None
        self.n_nodes: int = 0
        self.__post_init__()

        self._selected_terminal: int = None
    
    def __post_init__(self):
        pass

    @property
    def first_node(self) -> int:
        for i in range(1,self.n_nodes):
            if i not in self.nodes:
                return i
        return self.n_nodes
            
    def __update__(self):
        for node in self.nodes.values():
            self.network = node._parent
        
    def __validate__(self):
        # Check if number of nodes > 0
        if self.n_nodes == 0:
            raise ValueError("The component must have at least one node.")
        
        # Check if self.network is a Network object
        if not isinstance(self.network, Network):
            raise ValueError("The component must be connected to a Network object.")
        
        # Check if all nodes are Node objects with the same Network
        for node in self.nodes.values():
            if not isinstance(node, Node):
                raise ValueError(f"All nodes must be Node objects. Got {type(node)} instead.")
            if node._parent != self.network:
                raise ValueError("All nodes must belong to the same Network object.")
        
        # Check if the ground is defined, if its also a node and its network is the same network.
        if self.gnd is not None:
            if not isinstance(self.gnd, Node):
                raise ValueError("The ground must be a Node object.")
            if self.gnd._parent != self.network:
                raise ValueError("The ground node must belong to the same Network object.")

    def _check_full(self) -> None:
        """ Check if all nodes are connected, and if so, it calls __on_connect__(). """

        for i in range(1, self.n_nodes+1):
            if i not in self.nodes:
                return
        self.__on_connect__()

    def __on_connect__(self):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    def t(self, index: int) -> BaseComponent:
        self._selected_terminal = index
        return self
    
    def __lt__(self, other: Node) -> BaseComponent:
        if isinstance(other, Node):
            self.partial_connect(self._selected_terminal, other)
            return self
        return NotImplemented
    
    def __gt__(self, other: Node) -> BaseComponent:
        if isinstance(other, Node):
            self.partial_connect(self._selected_terminal, other)
            return self
        return NotImplemented
    
    def node(self, index: int) -> Node:
        if index not in self.nodes:
            self.nodes[index] = self.network.node()
        return self.nodes.get(index, None)

    def partial_connect(self, index: int, node: Node) -> BaseComponent:
        self.nodes[index] = node
        self.__update__()
        self._check_full()
        return self
    
    def connect(self, *nodes, gnd: Node = None) -> BaseComponent:
        """ Connect the component to the network. 

        Parameters:
        -----------
        *nodes : Node
            The nodes to connect the component to.
        gnd : Node
            The ground node to connect the component to.
        """
        self.gnd = gnd
        for i, node in enumerate(nodes):
            self.nodes[i+1] = node

        self.__update__()

        if self.gnd is None:
            self.gnd = self.network.gnd

        self.__validate__()
        self.__on_connect__()
        return self


class BaseTwoPort(BaseComponent):

    def __post_init__(self):
        self.n_nodes = 2

    def __lt__(self, other: Node) -> BaseComponent:
        if not isinstance(other, Node):
            return NotImplemented
        self.partial_connect(1, other)
        return self

    def __gt__(self, other: Node) -> Node:
        if not isinstance(other, Node):
            return NotImplemented
        self.partial_connect(2, other)
        return other
    
class BaseThreePort(BaseComponent):

    def __post_init__(self):
        self.n_nodes = 3

class BaseFourPort(BaseComponent):

    def __post_init__(self):
        self.n_nodes = 4