"""This script will demonstarte the basics of Heavi.

We will be making a simple 
"""

# First we import heavi

import heavi as hv

# Next we create our circuit. We can use the Network class
# but its better to use the more extended Model class

m = hv.Model()

# Heavi works by creating nodes and connecting components to them.
# The basic library of lumped elements allows us to directly connect them
# in the constructor of the components.
# More advanced components have extra methods. More about that later.

# We always have our ground node
gnd = m.gnd

# For this filter we will make two ports. To create a terminal
# for S-parameter analysis we need intermediate nodes and such thus we make this
# easier by using the build in methods!

node_in = m.node()
p1 = m.new_port(50, node_in)

# If we are lazy, we can also make the port and the node at the same time. 
node_out = m.quick_port(50)

# Our terminal object is now not available as before but we don't use those anyway.

# Next for our filter we need two intermediate nodes:
n1 = m.node()
n2 = m.node()

# Now we can connect some components
L1 = m.inductor(node_in, n1, 2.407e-9)
C1 = m.capacitor(n1, gnd, 1.66145e-12)
L2 = m.inductor(n1, n2, 5.0207e-9)
C2 = m.capacitor(n2, gnd, 1.55145e-12)
L3 = m.inductor(n2, node_out, 2.407e-9)

# Now that our modeling is done we can simply run our circuit!
fs = hv.frange(0.5e9, 8e9, 2001)
S = m.run_sp(fs)

# Our S-matrix object is an Sparameter object.

hv.plot_s_parameters(fs, [S.S11, S.S21, S.S12, S.S22], labels=['S11','S21','S12','S22'], logx=True)