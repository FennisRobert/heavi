from __future__ import annotations
from .rfcircuit import Network, Node

class BaseComponent:

    def __init__(self):
        self.network: Network = None
        self.nodes: dict[int, Node] = {}
        self.gnd: Node = None

    def __validate__(self):
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
        

    def __on_connect__(self):
        raise NotImplementedError("This method must be implemented in the child class.")
    
    def node(self, index: int) -> Node:
        return self.nodes.get(index, None)
    
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

        for node in self.nodes.values():
            if isinstance(node, Node):
                self.network = node._parent
        
        self.__validate__()
        self.__on_connect__()
        return self
