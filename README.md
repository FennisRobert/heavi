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

Install Heavi using pip (future work):

```bash
pip install heavi
```

For now, clone the repository and install locally:

```bash
git clone <repository-url>
cd heavi
pip install .
```

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
S = model.run_sparameter_analysis(f)

# Plot S-parameters
hf.plot_s_parameters(f, [S.S11, S.S21], labels=["S11", "S21"], linestyles=["-", "-"], colorcycle=[0, 1])

# Print all components in the model
model.print_components()
```

---

## API Reference

### `hf.Model()`
The central object for building and simulating circuits.

- **`node()`**: Creates a new circuit node.
- **`terminal(node, impedance)`**: Defines a terminal connected to a node with a given impedance.
- **`run_sparameter_analysis(frequencies)`**: Runs S-parameter analysis over the given frequency range.
- **`print_components()`**: Prints a list of all components in the model.

### `hf.lib.smd.SMDResistor(resistance, size)`
Creates an SMD resistor with specified resistance and size.

- **`connect(node1, node2)`**: Connects the resistor between two nodes.

### `model.filters.cauer_filter(gnd, input, output, center_freq, bandwidth, order, ripple, type, band_type)`
Adds a Cauer filter with specified parameters to the circuit.

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

## Future Work

- Support for nonlinear components (e.g., transistors, diodes).
- Time-domain simulations.
- Enhanced visualization capabilities.

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

Heavi draws inspiration from the pioneering work of circuit analysis and transmission line theory, honoring contributors like Oliver Heaviside and Gustav Kirchhoff.

