from __future__ import annotations
from .libgen import BaseComponent
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
from typing import Callable
import re
from loguru import logger

_FUNIT = {
    'hz': 1,
    'khz': 1000,
    'mhz': 1e6,
    'ghz': 1e9,
}
_NPORTS = {
    '.s1p': 1,
    '.s2p': 2,
    '.s3p': 3,
    '.s4p': 4,
    '.snp': None
}

def _ma_ri(mag: float, angle: float) -> complex:
    return mag * np.exp(1j * np.radians(angle))

def _db_ri(db: float, angle: float) -> complex:
    return 10**(db/20) * np.exp(1j * np.radians(angle))

def _ri_ri(real: float, imag: float) -> complex:
    return real + 1j * imag

_DATAMAP = {
    'ma': _ma_ri,
    'db': _db_ri,
    'ri': _ri_ri
}

class FileBasedNPort(BaseComponent):

    def __init__(self, filename: str,
                 n_ports: int = None,
                 ignore_extension: bool = False,
                 interp_type: str = 'cubic'):
        super().__init__()
        path = Path(filename)
        if not path.exists():
            raise ValueError(f'The provided filename {path} does not exist.')

        if path.suffix not in ('.s1p','.s2p','.s3p','.s4p','.snp','.ts') and not ignore_extension:
            raise ValueError(f'Can only open touchstone files, files with {path.suffix} are not a valid touchstone extension.')
        
        with open(filename,'r') as f:
            self._lines = f.read().split('\n')
        
        self.Sdata = None

        self.freq_unit: float = 1e9
        self.param_type: str = 's'
        self.data_type: str = 'ma'
        self._converter: Callable = _DATAMAP['ma']
        self.refimp: float = 50
        self.n_ports: int = None
        self.interp_type: str = interp_type
        self.Sdata: np.ndarray = None
        self.Fdata: np.ndarray = None

        if self.param_type != 's':
            raise ValueError('Only S-parameter touchstone files are supported as of this moment.')

        self._parse_touchstone()
    
    def summarize_data(self):
        print(f'Frequency unit: {self.freq_unit}')
        print(f'Parameter type: {self.param_type}')
        print(f'Data type: {self.data_type}')
        print(f'Converter: {self._converter}')
        print(f'Reference impedance: {self.refimp}')
        print(f'Number of ports: {self.n_ports}')

    def _parse_touchstone(self) -> None:
        s_data = []
        f_data = []
        s_collector = []
        freq = None
        for line in self._lines:
            if not line:
                continue
            
            stripped = line.strip().lower()
            # Comment line
            if stripped[0]=='!':
                continue
            
            # Options line
            if stripped[0]=='#':
                funit = re.findall(r'(hz|khz|mhz|ghz)', stripped)
                param_type = re.findall(r'([syzgh])', stripped)
                data_type = re.findall(r'(ma|db|ri)', stripped)
                refimp = re.findall(r'r\s+(\d+)', stripped)

                if funit:
                    self.freq_unit = _FUNIT[funit[0]]
                if param_type:
                    self.param_type = param_type[0]
                if data_type:
                    self.data_type = data_type[0]
                    self._converter = _DATAMAP[data_type[0]]
                if refimp:
                    self.refimp = float(refimp[0])
            
                continue
            
            # Version 2.0 Keyword line:
            if stripped[0]=='[':
                logger.warning('Version 2.0 touchstone files are not supported yet and keywords will be ignored')
                continue
            # Data line
            # remove the exclamation mark plus all symbols behind it if it occurs
            line = line.split('!')[0]

            nums = [float(x) for x in line.split(' ') if x]

            if len(nums)%2 == 1:
                if len(s_collector) > 0:
                    s_data.append(s_collector)
                    f_data.append(freq)
                s_collector = nums[1:]
                freq = nums[0]
            else:
                s_collector.extend(nums)
            
        s_data.append(s_collector)
        f_data.append(freq)

        ssample = s_data[0]
        self.n_ports = int(np.sqrt(len(ssample) // 2))
        self.n_nodes = self.n_ports

        self.n_frequencies = len(f_data)
        self.Fdata = np.array(f_data) * self.freq_unit
        self.Sdata = np.zeros((self.n_frequencies, self.n_ports, self.n_ports), dtype=complex)

        for i, s in enumerate(s_data):
            sdata = [self._converter(s1,s2) for s1,s2 in zip(s[::2], s[1::2])]
            Smat = np.array(sdata).reshape(self.n_ports, self.n_ports)
            if self.n_ports == 2:
                Smat = Smat.T
            self.Sdata[i,:,:] = Smat

    def renormalize(self, new_impedance: float) -> FileBasedNPort:
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

    def __on_connect__(self):
        s_functions = []
        for i in range(self.n_ports):
            row = []
            for j in range(self.n_ports):
                row.append(interp1d(self.Fdata, self.Sdata[:,i,j], kind=self.interp_type, bounds_error=False, fill_value=(self.Sdata[0,i,j],self.Sdata[-1,i,j])))
            s_functions.append(row)
        

        self.network.n_port_S(self.network.gnd, 
                              [self.nodes[i] for i in range(1,self.n_ports+1)],s_functions,self.refimp)