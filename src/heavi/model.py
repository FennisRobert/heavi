from .rfcircuit import Network, Node
from .filtering import FilterType, BandType, CauerType, Filtering


class Model(Network):

    def __init__(self, default_name: str = "Node", 
                 filter_library: Filtering = Filtering, 
                 suppress_loadbar: bool = False):
        super().__init__(default_name, suppress_loadbar=suppress_loadbar)
        self.filters: Filtering = filter_library(self)
        self.numbered_nodes: dict[int, Node] = dict()

    def __call__(self, index: int) -> Node:
        if index not in self.numbered_nodes:
            self.numbered_nodes[index] = self.node()
        node = self.numbered_nodes[index]
        node._parent = self
        return node

class QuickModel(Network):

    def __init__(self, default_name: str = "Node", 
                 filter_library: Filtering = Filtering, 
                 suppress_loadbar: bool = False):
        super().__init__(default_name, suppress_loadbar=suppress_loadbar)
        self.filters: Filtering = filter_library(self)

    def LC(self, n1: Node, n2: Node, L: float, C: float) -> None:
        self.capacitor(n1, n2, C)
        self.inductor(n1, n2, L)
    
    def LC_series(self, n1: Node, n2: Node, L: float, C: float) -> None:
        nmid = self.node()
        self.capacitor(n1, nmid, C)
        self.inductor(nmid, n2, L)

    def C(self, n1: Node, n2: Node, C: float) -> None:
        self.capacitor(n1, n2, C)
    
    def L(self, n1: Node, n2: Node, L: float) -> None:
        self.inductor(n1, n2, L)
    