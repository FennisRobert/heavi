from .rfcircuit import Network, Node
from .filtering import FilterType, BandType, CauerType, Filtering
from .library import Library

class Model(Network):

    def __init__(self, default_name: str = "Node", 
                 filter_library: Filtering = Filtering, 
                 component_library: Library = Library,
                 suppress_loadbar: bool = False):
        super().__init__(default_name, suppress_loadbar=suppress_loadbar)
        self.filters: Filtering = filter_library(self)
        self.lib: Library = component_library(self)