# Heavi

Heavi is a Python-based circuit simulation library designed for efficient modeling and analysis of linear components. It allows users to build circuits programmatically, run analyses (such as S-parameters), and visualize results. Currently, the library supports only linear components but provides a foundation for advanced circuit simulations.

## Features

- **Linear Component Simulation**: Model circuits with resistors, capacitors, and inductors.
- **Custom Filters**: Create filters (e.g., Cauer, Chebyshev) with ease.
- **S-Parameter Analysis**: Perform S-parameter analysis on two-port networks.
- **Visualization Tools**: Plot S-parameters with customizable options.
- **Programmatic Circuit Building**: Construct circuits directly in Python for maximum flexibility.

---

## Installation

For now, clone the repository and install locally:

```bash
git clone https://github.com/FennisRobert/heavi.git
cd heavi
pip install .
```

Optionally, its easier to just download the 
---

## Getting Started

Here is an example that demonstrates building a simple two-port circuit with a 4th-order bandpass filter, running S-parameter analysis, and plotting the results.

```python
import heavi as hf

# Create a new circuit model
model = hf.Model()

# Define circuit nodes and terminals
n1 = model.node()
p1 = model.terminal(n1, 50)  # Terminal 1 with 50-ohm impedance
n2 = model.node()
n3 = model.node()
p2 = model.terminal(n3, 50)  # Terminal 2 with 50-ohm impedance

# Add a resistor between nodes n2 and n3
resistor = hf.lib.smd.SMDResistor(5, hf.lib.smd.SMDResistorSize.R0402).connect(n2, n3)

# Add a 4th-order bandpass filter between ground and node n1, connecting to n2
model.filters.cauer_filter(
    model.gnd, n1, n2, 2e9, 70e6, 5, 0.03, hf.FilterType.CHEBYCHEV, type=hf.BandType.BANDPASS
)

# Define the frequency range for S-parameter analysis
f = hf.frange(1.8e9, 2.2e9, 2001)

# Perform S-parameter analysis
S = model.run(f)

# Plot S-parameters
hf.plot_s_parameters(f, [S.S11, S.S21], labels=["S11", "S21"], linestyles=["-", "-"], colorcycle=[0, 1])

# Print all components in the model
model.print_components()
```

---

## API Reference

### heavi.Model
The `Model` class is the central interface for building and simulating circuits. It inherits from the `Network` class and has additional libraries for component creation.

#### **Initialization**
```python
Model(default_name: str = "Node", filter_library: Filtering = Filtering, component_library: Library = Library)
```
- **`default_name`**: Default prefix for node names.
- **`filter_library`**: Library for filter definitions.
- **`component_library`**: Library for circuit components.

#### **Key Methods**
Most of the methods mentioned below are defined as `Network` methods.

- **`node(name: str = None) -> Node`**
  Creates a new circuit node.

- **`terminal(signal_node: Node, Z0: float, gnd_node: Node = None) -> Terminal`**
  Adds a terminal to the circuit with a specified impedance.

- **`resistor(node1: Node, node2: Node, R: float) -> Component`**
  Adds a resistor between two nodes.

- **`capacitor(node1: Node, node2: Node, C: float) -> Component`**
  Adds a capacitor between two nodes.

- **`inductor(node1: Node, node2: Node, L: float) -> Component`**
  Adds an inductor between two nodes.

- **`transmissionline(gnd: Node, port1: Node, port2: Node, Z0: float, er: float, L: float) -> Component`**
  Adds a transmission line with specified properties.

- **`run(frequencies: np.ndarray) -> Sparameters`**
  Runs S-parameter analysis for the specified frequency range.
---

### heavi.Network
The `Network` class provides foundational methods for building and analyzing circuits.

#### **Initialization**
```python
Network(default_name: str = 'Node')
```
- **`default_name`**: Default prefix for node names.

---

### heavi.plot_s_parameters
The `plot_s_parameters` plot function uses `matplotlib` and offers a convenient means of displaying S-parameters.
```python
plot_s_parameters(frequencies: np.ndarray, S_parameters: list[np.ndarray], labels: list[str], linestyles: list[str], colorcycle: list[int])
```
Plots S-parameter data.

#### Parameters:
- **`frequencies`**: Frequency range in Hz.
- **`S_parameters`**: List of S-parameter arrays.
- **`labels`**: Labels for each plot line.
- **`linestyles`**: Line styles for each plot line.
- **`colorcycle`**: List of color indices for the plot.

---

### heavi.sparam.Sparameters
The `Sparameters` class encapsulates the results of S-parameter analysis and mostly just contains the S-parameters as a 3-dimensional numpy array with the first two dimensions being the S-parameter matrix and the last the frequency axis.

#### **Attributes**
- **`S`**: S-parameter matrix.
- **`frequencies`**: Frequency range associated with the analysis.

#### **Properties**
All possible S-parameters up to a 5-port network are predefined as properties for quick access and auto-completion purposes.
- **`S11`**: The S11 array data.
- **`S12`**: The S12 array data.
- **`S55`**: The S55 array data.

#### **Key Methods**
Most of the methods mentioned below are defined as `Network` methods.

- **`S(i: int, j: int) -> np.ndarray`**
  Returns the Sij component of the S-matrix.

---

For more advanced usage, refer to the inline documentation in the `heavi` package or the source code.

---

## Visualization

The `hf.plot_s_parameters()` function provides an easy way to visualize S-parameter data:

```python
hf.plot_s_parameters(frequencies, [S11, S21], labels=["S11", "S21"], linestyles=["-", "--"], colorcycle=[0, 1])
```

- **`frequencies`**: Frequency range (Hz).
- **`S-parameters`**: A list of S-parameters to plot.
- **`labels`**: Labels for each plot line.
- **`linestyles`**: Line styles for each plot.
- **`colorcycle`**: Color indices for the lines.

---
## Features

- Stochastic components and system modeling. 
- Most common components, impedance, admittance, port, capacitor, inductor, transmission line and N-port S-parameters.
- BaseCompont class for custom library creation
- Cauer filter design automization (Low, high and bandpass Chebychev and Maximally flat).
- Impedance transformers (Chebychev only)

## Future features

- Touchstone import
- Advanced PCB Router
- Larger library of components and predefined interfaces/functions
- Support for nonlinear components (e.g., transistors, diodes).
- Time-domain simulations.
- Enhanced visualization capabilities.

---

## Contributing

Contributing options come later. Please reach out via my website: www.emerge-tools.org

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


