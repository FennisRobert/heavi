########################################################################################
##
##    The model module contains the Model and QuickModel classes.
##    An extended version of the Network class, the Model class is used to create
##
##    Author: Robert Fennis
##    Date: 2025
##
########################################################################################

#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------


from .network import Network, Node
from .filtering import FilterType, BandType, CauerType, Filtering

#  __             __   __      __   ___  ___          ___    __        __  
# /  ` |     /\  /__` /__`    |  \ |__  |__  | |\ | |  |  | /  \ |\ | /__` 
# \__, |___ /~~\ .__/ .__/    |__/ |___ |    | | \| |  |  | \__/ | \| .__/ 
# -------------------------------------------


class Model(Network):
    """The Model class is an extension of the Network class. It can be used instead of the Network
    class and offers additional functionality for creating circuits.
    """    
    def __init__(self, default_name: str = "Node", 
                 filter_library: Filtering = Filtering, 
                 suppress_loadbar: bool = False):
        """Initializes the Model object.

        Args:
            default_name (str, optional): Default node name prefix. Defaults to "Node".
            filter_library (Filtering, optional): The filter library to use. Defaults to Filtering.
            suppress_loadbar (bool, optional): Whether or not to show numba loadbars. Defaults to False.
        """        
        super().__init__(default_name, suppress_loadbar=suppress_loadbar)
        self.filters: Filtering = filter_library(self)
        self.numbered_nodes: dict[int, Node] = dict()

    def __call__(self, index: int) -> Node:
        if index not in self.numbered_nodes:
            self.numbered_nodes[index] = self.node()
        node = self.numbered_nodes[index]
        node._parent = self
        return node
    
    