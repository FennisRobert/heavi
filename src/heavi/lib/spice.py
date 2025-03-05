from __future__ import annotations

from pathlib import Path
from .libgen import BaseTwoPort
from ..rfcircuit import Node
import re
from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

####### Constants #######

_POWERS = {
        "f": -15,
        "p": -12,
        "n": -9,
        "u": -6,
        "m": -3,
        "k": 3,
        "q": 6,
        "g": 9,
        "t": 12,
    }

_COMPONENT_TEMPLATE: str = '''


class #NAME#(BaseTwoPort):
    """ #DESCRIPTION# """
    def __init__(self):
        super().__init__()
    
    def __on_connect__(self):
        node1 = self.node(1)
        node2 = self.node(2)
        
#NODES#
        
#REST#'''

_PYTHON_BASE_TEMPLATE = '''
from __future__ import annotations
from heavi.lib.libgen import BaseTwoPort
from heavi.rfcircuit import Node


'''

# Define basic component classes.

class SpaceCompContainer:
    _implemented: bool = False
    def __init__(self, name: str, nodes: List[int], value: float):
        self.name = name
        self.nodes = nodes
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, nodes={self.nodes}, value={self.value})"
    
class SpiceR(SpaceCompContainer):
    _implemented: bool = True

class SpiceL(SpaceCompContainer):
    _implemented: bool = True

class SpiceC(SpaceCompContainer):
    _implemented: bool = True

class SpiceV(SpaceCompContainer):
    _implemented: bool = False

class SpiceI(SpaceCompContainer):
    _implemented: bool = False

class SpiceX(SpaceCompContainer):
    _implemented: bool = False

class SpiceD(SpaceCompContainer):
    _implemented: bool = False

class Subcircuit:

    def __init__(self, name: str, nodes: List[int], components: List[SpaceCompContainer]):
        self.name = name
        self.nodes = nodes
        self.components = components
        
    def __repr__(self):
        return f"Subcircuit(name={self.name}, nodes={self.nodes}, components={self.components})"
    
    @property
    def is_supported(self) -> bool:
        ''' Returns True if all components in the subcircuit are supported by the compiler. '''
        for component in self.components:
            if not component._implemented:
                return False
        return True
    
    def all_containing_nodes(self) -> list[int]:
        ''' Returns a list of all nodes mentioned in the subcircuit. '''
        nodes = self.nodes
        for component in self.components:
            nodes.extend(component.nodes)
        return sorted(list(set(nodes)))


class SpiceASY:

    def __init__(self, description: list[str], spicemodel: str, instname: str, model_file: str):
        self.description = description
        self.spicemodel = spicemodel
        self.instname = instname
        self.model_file = model_file

        if self.instname is None:
            self.instname = 'unnamed'
        if self.model_file is None:
            self.model_file = 'unnamed'
    

def _parse_asy_file(file_path: str) -> SpiceASY:
    ''' Parses a .asy file and returns a list of all components. 
    
    parses the following datalines:
    SYMATTR Description WE-MPSA EMI Multilayer Power Suppression Bead \nAfter inserting, right-click on the symbol to select the part number. \nwww.we-online.com/catalog/WE-MPSA \n\nPlease note disclaimer in WE-MPSA.lib.
    SYMATTR SpiceModel 0805_78279220321_320Ohm
    SYMATTR Prefix x
    SYMATTR InstName L
    SYMATTR ModelFile WE-MPSA.lib
    '''

    
    components = []
    description = None
    spicemodel = None
    instname = None
    model_file = None

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("SYMATTR Description"):
                description = line.split("SYMATTR Description ")[1].strip().replace("\\n", "\n")
            elif line.startswith("SYMATTR SpiceModel"):
                spicemodel = line.split("SYMATTR SpiceModel ")[1].strip()
            elif line.startswith("SYMATTR InstName"):
                instname = line.split("SYMATTR InstName ")[1].strip()
            elif line.startswith("SYMATTR ModelFile"):
                model_file = line.split("SYMATTR ModelFile ")[1].strip()
    
    return SpiceASY(description, spicemodel, instname, model_file)
              

def merge_continuation_lines(file_path: str) -> List[str]:
    """
    Reads the file and merges lines that begin with '+' with the previous line.
    """
    merged_lines = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        current_line = ""
        for line in f:
            # Remove trailing newline and spaces.
            line = line.rstrip()
            # If the line starts with '+' it is a continuation of the previous line.
            if line.startswith("+"):
                current_line += " " + line.lstrip("+").strip()
            else:
                # If we have a previous line, add it to the list.
                if current_line:
                    merged_lines.append(current_line)
                current_line = line.strip()
        if current_line:
            merged_lines.append(current_line)
    return merged_lines




def parse_token(string: str, numdict: dict) -> float:
    ''' Parses a string with a number and a power of 10. '''
    if string[0] == "{" and string[-1] == "}":
        return numdict.get(string[1:-1], None)
    else:
        string = re.sub(r'(?i)\bmeg\b', 'M', string)
        groups = re.match(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)([a-zA-Z]?)([a-zA-Z]*)?', string)
        # Groups is Number, potential power and unit. Units are ignored
        if groups is None:
            raise ValueError(f"Invalid number: {string}, {numdict}")
        number = float(groups.group(1))
        power = groups.group(2)
        if power:
            power = _POWERS.get(power.lower(), 0)
            number *= 10 ** power
        return number

def parse_subckt_line(line: str) -> Subcircuit:
    # Remove excess spaces and normalize whitespace
    line = re.sub(r"\s+", " ", line.strip()).upper()

    # Match the .subckt line structure
    match = re.match(r"\.SUBCKT\s+([a-zA-Z\d\-\_\.\+]+)\s+([0-9a-zA-Z\s]+)(?:\s+PARAMS:\s*(.*))?", line)
    if not match:
        raise ValueError(f"Invalid .subckt line: {line}")
    name = match.group(1)  # Subcircuit name
    nodes = list(match.group(2).split()) # Convert nodes to integers
    params_str = match.group(3) if match.group(3) else ""  # Parameter string

    # Extract parameters using regex (matches key=value pairs)
    params = {}
    param_matches = re.findall(r"(\w+)=([\d.eE+-]+[a-zA-Z]?)", params_str)

    

    for key, value in param_matches:
        params[key] = parse_token(value,{}) # Direct conversion

    return name, nodes, params

def parse_spice_lib(file_path: str) -> List[Subcircuit]:
    """
    Parses a SPICE .lib file (with or without continuation lines)
    and returns a list of Subcircuit objects ONLY if the subcircuit contains
    components whose names begin with R, L, C, V, or I.
    
    Each Subcircuit object contains:
      - name: the subcircuit name (from the .subckt header)
      - nodes: the list of nodes declared in the header
      - components: a list of component objects (SpiceR, SpiceL, SpiceC, etc.)
    
    If any component in the subcircuit does not begin with one of the allowed
    letters, that entire subcircuit is skipped.
    """
    allowed_types = {"R", "L", "C", "V", "I"}
    merged_lines = merge_continuation_lines(file_path)
    subcircuits: List[Subcircuit] = []
    current_subckt: Optional[str] = None
    current_nodes: List[str] = []
    current_components: List[SpiceComponent] = []
    valid_subckt = True  # Flag to indicate that current subckt contains only allowed components
    
    params = {}
    for iline, line in enumerate(merged_lines, start=1):
        # Skip comment lines.
        if line.startswith("*") or not line:
            continue
        
        lower_line = line.lower()
        if lower_line.startswith(".subckt"):
            # If we were in a subckt (without seeing .ends) we could handle it here.
            current_subckt, current_nodes, params = parse_subckt_line(line)
            # tokens[0] is ".subckt", tokens[1] is the subckt name, tokens[2:] are node names.
            valid_subckt = True  # start with a valid subcircuit assumption
        elif lower_line.startswith(".ends"):
            if current_subckt is not None and valid_subckt:
                subcircuit_obj = Subcircuit(name=current_subckt,
                                             nodes=current_nodes,
                                             components=current_components)
                subcircuits.append(subcircuit_obj)
            # Reset for the next subcircuit.
            current_subckt = None
            current_nodes = []
            current_components = []
            valid_subckt = True
        else:
            # We are inside a subcircuit definition.
            if current_subckt is not None:

                #removed_statements = re.sub(r"\{.*?\}", "", line)
                removed_statements = re.sub(r"\{\w+=\w+\}", "", line)
                tokens = [token for token in removed_statements.split() if token]
                # Expect at least 4 tokens: component name, node1, node2, value.
                if len(tokens) < 4:
                    logger.error(f"Error parsing component: {line} on line number {iline} in file {file_path}. Insufficient information detected")
                    continue  # Or optionally raise an error.
                
                comp_name = tokens[0]
                comp_type = comp_name[0].upper()
                
                node1 = tokens[1]
                node2 = tokens[2]

                if comp_type not in allowed_types:
                    valid_subckt = False
                    break
                
                try:
                    value = parse_token(tokens[3], params)
                except ValueError:
                    logger.error(f"Error parsing component: {line} on line number {iline} in file {file_path}")
                    valid_subckt = False
                    break
                
                # Create the appropriate component instance.
                try:
                    if comp_type == "R":
                        comp_obj = SpiceR(name=comp_name, nodes=[node1, node2], value=value)
                    elif comp_type == "L":
                        comp_obj = SpiceL(name=comp_name, nodes=[node1, node2], value=value)
                    elif comp_type == "C":
                        comp_obj = SpiceC(name=comp_name, nodes=[node1, node2], value=value)
                    elif comp_type == "V":
                        comp_obj = SpiceV(name=comp_name, nodes=[node1, node2], value=value)
                    elif comp_type == "I":
                        comp_obj = SpiceI(name=comp_name, nodes=[node1, node2], value=value)
                    elif comp_type == "X":
                        comp_obj = SpiceX(name=comp_name, nodes=[node1, node2], value=value)
                    elif comp_type == "D":
                        comp_obj = SpiceD(name=comp_name, nodes=[node1, node2], value=value)
                    else:
                        # Should not get here because of the allowed_types check.
                        valid_subckt = False
                        continue
                except ValueError:
                    logger.error(f"Error parsing component: {line} on line number {iline} in file {file_path}")
                    valid_subckt = False
                    continue
                current_components.append(comp_obj)
    return subcircuits

class SpiceComponent(BaseTwoPort):

    def __init__(self, subcircuit: Subcircuit):
        super().__init__()
        self.subcircuit = subcircuit

    def add_component(self, component: SpaceCompContainer, node1: Node, node2: Node):
        value = component.value
        if isinstance(component, SpiceR):
            self.network.resistor(node1, node2, value)
        elif isinstance(component, SpiceL):
            self.network.inductor(node1, node2, value)
        elif isinstance(component, SpiceC):
            self.network.capacitor(node1, node2, value)
        elif isinstance(component, SpiceV):
            self.network.voltage_source(node1, node2, value)
        elif isinstance(component, SpiceI):
            self.network.current_source(node1, node2, value)

    def __on_connect__(self):
        node1 = self.node(1)
        node2 = self.node(2)

        mentioned_nodes = self.subcircuit.all_containing_nodes()

        node_map = {self.subcircuit.nodes[0]: node1, self.subcircuit.nodes[1]: node2}
        for i in mentioned_nodes:
            if i not in node_map:
                node_map[i] = self.network.node(f'{self.subcircuit.name}[{i}]')
        
        for component in self.subcircuit.components:
            node1 = node_map[component.nodes[0]]
            node2 = node_map[component.nodes[1]]
            print(f"Adding component {component.name} with value {component.value} between nodes {node1} and {node2}")
            self.add_component(component, node1, node2)


class SpiceLibrary:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.subcircuits: list[Subcircuit] = parse_spice_lib(filepath)
        self.circuit_dict: dict[str, Subcircuit] = {subckt.name.lower(): subckt for subckt in self.subcircuits}
    
    def merge(self, other: SpiceLibrary) -> SpiceLibrary:
        ''' Merge another SpiceLibrary objects into the current one. '''
        self.subcircuits.extend(other.subcircuits)
        self.circuit_dict.update(other.circuit_dict)
        return self
    
    def __call__(self, searchterm: str) -> Optional[SpiceComponent]:
        sbk = self.find(searchterm)
        if sbk is None:
            return None
        return SpiceComponent(sbk)
    
    def find(self, name: str) -> Optional[Subcircuit]:
        results = []
        name = name.lower()

        if name in self.circuit_dict:
            return self.circuit_dict[name]
        else:
            logger.warning(f"No subcircuits found for '{name}'")
            return None


def import_spice_library_directory(directory: str, libfilename: str = None) -> None:
    ''' This function searches all .lib files in the provided directory to build one big SpiceLibrary object.
    Then it searches foor all spice .asy files to generate a set of components. 
    Based on this it generates a Python file with the components and the SpiceLibrary object. '''
    
    lib_files = Path(directory).rglob("*.lib")
    asy_files = Path(directory).rglob("*.asy")

    spicelib = SpiceLibrary(next(lib_files))

    for file in lib_files:
        spicelib.merge(SpiceLibrary(file))
    
    asys = [_parse_asy_file(file) for file in asy_files]

    pythonlines = _PYTHON_BASE_TEMPLATE

    for asy in asys:
        sbk = spicelib.find(asy.spicemodel)

        if sbk is None:
            # Log warning already thrown in the .find function
            continue

        if not sbk.is_supported:
            logger.warning(f"Subcircuit '{sbk.name}' contains unsupported components. Skipping this in outputting.")
            continue
        
        sbknode1 = sbk.nodes[0]
        sbknode2 = sbk.nodes[1]
        
        # Generate the component class name by removing illegal python characters.
        component_class_name = asy.instname.upper() + sbk.name.replace(" ", "").replace('.','p').replace('-','_')

        #Substitute the class name
        component_text = _COMPONENT_TEMPLATE.replace("#NAME#", component_class_name)
        
        #Substitute the description
        component_text = component_text.replace("#DESCRIPTION#", f"Component for {asy.spicemodel}\n{asy.description}")
        
        mentioned_nodes = sbk.all_containing_nodes()
        
        node_text = []
        # Define node statements for each node in the subcircuit that isn't also the main subcircuit connectio nodes.
        for node in mentioned_nodes:
            if node == sbknode1 or node == sbknode2:
                continue
            node_text.append(f"        N_{node} = self.network.node('{sbk.name}[{node}]')")
        component_text = component_text.replace("#NODES#", "\n".join(node_text))

        # Define the component statements for each component in the subcircuit.
        component_lines = []
        for component in sbk.components:
            # Define the cubcircuit component nodes
            node1 = component.nodes[0]
            node2 = component.nodes[1]

            nn1 = f"N_{node1}"
            nn2 = f"N_{node2}"
            
            # Replace the nodes by the two primary nodes of the subcircuit
            if node1 == sbknode1:
                nn1 = "node1"
            elif node1 == sbknode2:
                nn1 = "node2"
            if node2 == sbknode1:
                nn2 = "node1"
            elif node2 == sbknode2:
                nn2 = "node2"

            fname = ''
            if isinstance(component, SpiceR):
                fname = 'resistor'
            elif isinstance(component, SpiceL):
                fname = 'inductor'
            elif isinstance(component, SpiceC):
                fname = 'capacitor'

            component_lines.append(f"        self.network.{fname}({nn1}, {nn2}, {component.value})")
        
        component_text = component_text.replace("#REST#", "\n".join(component_lines))
        pythonlines += component_text

        with open(f"{libfilename}.py", "w") as f:
            f.write(pythonlines)