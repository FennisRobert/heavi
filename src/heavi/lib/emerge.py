
#          __   __   __  ___  __  
# |  |\/| |__) /  \ |__)  |  /__` 
# |  |  | |    \__/ |  \  |  .__/ 
# -------------------------------------------


from __future__ import annotations
from .libgen import SubCircuit
import numpy as np
from scipy.interpolate import interp1d


class EMergeModel(SubCircuit):
    """An EMerge N-port model

    """
    def __init__(self, emerge_data: tuple[np.ndarray, np.ndarray]
                 ,interp_type: str = 'cubic'):
        super().__init__()
        
        self.Fdata, self.Sdata = emerge_data
        
        self.n_ports: int = self.Sdata.shape[1]
        self.n_nodes = self.n_ports
        self.interp_type: str = interp_type
        self.refimp: float = 50
        
    def renormalize(self, new_impedance: float) -> EMergeModel:
        """ Renormalize the touchstone file to a new impedance.

        Args:
            new_impedance (float): The new impedance to normalize the touchstone file to.

        Returns:
            FileBasedNPort: The renormalized touchstone file subcircuit.
        """        
        Zold = self.refimp
        Znew = new_impedance
        A = np.eye(self.n_ports, self.n_ports) * np.sqrt(Znew/Zold)*(1/(Znew+Zold))
        R = np.eye(self.n_ports, self.n_ports) * ((Znew-Zold)/(Znew+Zold))
        iA = np.linalg.pinv(A)
        E = np.eye(self.n_ports, self.n_ports)
        for iif in range(self.n_frequencies):
            S = self.Sdata[iif,:,:]
            self.Sdata[iif,:,:] = iA @ (S - R) @ np.linalg.pinv(E - R @ S) @ A
        self.refimp = new_impedance
        return self

    def __on_connect__(self):
        s_functions = []
        for i in range(self.n_ports):
            row = []
            for j in range(self.n_ports):
                row.append(interp1d(self.Fdata, self.Sdata[:,i,j], kind=self.interp_type, bounds_error=False, fill_value=(self.Sdata[0,i,j],self.Sdata[-1,i,j])))
            s_functions.append(row)
        

        self.network.n_port_S(self.network.gnd, 
                              [self.nodes[i] for i in range(1,self.n_ports+1)],s_functions,self.refimp)
        