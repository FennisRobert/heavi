########################################################################################
##
##    Node Class
##    This file contains the Node class, which is used to represent nodes in the network.
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
from dataclasses import dataclass
from typing import Any

#  __             __   __   ___  __  
# /  ` |     /\  /__` /__` |__  /__` 
# \__, |___ /~~\ .__/ .__/ |___ .__/ 
# -------------------------------------------


class UnassignedType:

    def __str__(self):
        return '_'
    
    def __repr__(self):
        return '_'
    
    def __eq__(self, other):
        if other is None:
            return True
        return False

UNASSIGNED = UnassignedType()

@dataclass
class Node:
    """ Node class for the Network object. """
    name: str
    _index: int = UNASSIGNED
    _parent: Any = None
    _linked: Node = None
    _gnd: bool = False

    def __repr__(self) -> str:
        if self._gnd:
            return 'Node_GND'
        if self._linked is None:
            return f"{self.name}[{self._index}]"
        else:
            return f"LinkedNode[{self._linked}]"
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def __hash__(self):
        return hash(f'{self.name}_{self.index}')
    
    def set_index(self, index: int):
        """Assigns an index to the node. Should only be used by the Network object.

        Args:
            index (int): The node index.
        """        
        self._index = index

    def unique(self) -> Node:
        """Unique returns the unique variant of the node. If the node is linked to another node, 
        the linked node is returned. If the node is not linked, the node itself is returned.

        Returns:
            Node: The unique node.
        """        
        if self._linked is not None:
            return self._linked
        return self
    
    @property
    def index(self) -> int:
        """Returns the index of the node. If the node is linked to another node, 
        the index of the linked node is returned.

        Returns:
            int: the node index
        """        
        if self._linked is not None:
            return self._linked.index
        return self._index

    def merge(self, other: Node) -> Node:
        """Merges the current node with another node. The node that is merged with is returned."""
        self._linked = other
        return self
    
    def __gt__(self, other: Node) -> Node:
        """Merges the current node with another node. The node that is merged with is returned."""
        if isinstance(other, Node):
            self._linked = other
            return other
        return NotImplemented


@dataclass
class MatSource:
    """The MatSource class is used to represent a source in the network.
    This class is used only to store indices of required extra source objects during
    Matrix construction. Each MNA matrix requires additional collumns and rows for each
    Source object. Components that require additional sources to represent their behavior
    should return a MatSource object with the required indices during the matrix construction
    """    
    index: int

class NodeAllocator:

    def __init__(self, index: str, parent):
        self._index: str = index
        self._pointer: int = 0
        self._nodes: dict[int, Node] = dict()
        self._parent = parent
    
    def _get_new_number(self) -> int:
        """Returns the first available number for a new node.

        Returns:
            int: The new node number
        """        
        number = self._pointer
        while True:
            self._pointer += 1
            if self._pointer not in self._nodes:
                break
        return number
    
    def _make(self, number: int) -> Node:
        """Creates a node with the specified number.

        Args:
            number (int): the index of the node

        Returns:
            Node: The new node object
        """        
        node_obj = Node(f'{self._index}{number}',_parent = self._parent)
        self._nodes[number] = node_obj
        return node_obj
                    
    def request(self, number: int = None) -> Node:
        """Requests a node with the specified number. If the node does not exist, it is created.
        If no number is specified, a new number is generated. If the node exists, it is returned.

        Args:
            number (int, optional): The node index. Defaults to None.
        
        Returns:
            Node: The requested node object.
        """
        new = False
        if number is None:
            number = self._get_new_number()
        if number not in self._nodes:
            self._make(number)
            new = True
        return self._nodes[number], new