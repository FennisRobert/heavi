"""
In this temo we will look at how to import a touchstone file.

We use the touchstone file as was given on the Microwave101 website: https://www.microwaves101.com/encyclopedias/snp-format

"""

import heavi as hv
from pathlib import Path

# We generate our Model object
m = hv.Model()

# We get our Ground node
gnd = m.gnd
# We create two ports. This quick_port() method automatically returns the two nodes.

n1 = m.quick_port(50)
n2 = m.quick_port(50)

# Because I don't know where you'll put this Python file and the s2p file in the examples folder, this will always look in the directory that this Python file is in.
step_path = Path(__file__).parent / 'cha3024_99f_lna.s2p' 

# We can simply create our LNA object as a File Based N-port network.
LNA = hv.FileBasedNPort(str(step_path), 2)

# To connect our LNA to the port nodes we sumply call the Connect method.
LNA.connect(n1, n2) # Here we add the nodes that we want to connect our LNA to in order of the ports, port1, port2 etc.

# And we run!
fs = hv.frange(1e9, 20e9, 2001)

S = m.run_sp(fs)

hv.plot_s_parameters(fs, [S.S11, S.S21, S.S12, S.S22], labels=['S11','S21','S12','S22'])