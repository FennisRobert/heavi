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


@dataclass
class Node:
    """ Node class for the Network object. """
    name: str
    _index: int = None
    _parent: Any = None
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
class MatSource:
    """The MatSource class is used to represent a source in the network.
    This class is used only to store indices of required extra source objects during
    Matrix construction. Each MNA matrix requires additional collumns and rows for each
    Source object. Components that require additional sources to represent their behavior
    should return a MatSource object with the required indices during the matrix construction
    """    
    index: int
