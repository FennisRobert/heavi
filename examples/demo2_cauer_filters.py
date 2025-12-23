"""
In This example we will look at creating Cauer filters automatically with the Cauer filter designer in Heavi.
"""

import heavi as hv
import numpy as np

np.set_printoptions(precision=2, linewidth=200)

# We again create our model and nodes
model = hv.Model()

n1 = model.quick_port(50)
n2 = model.quick_port(50)

# In the Model class we have some common filter design features.
# We will create a band-pass filter at 2.5GHz with a bandwidth of 0.2GHz. 5th order with 0.05dB Ripple.
model.filters.cauer_filter(model.gnd, n1, n2, 2.5e9, 0.2e9, 5, 0.05, hv.FilterType.CHEBYCHEV, hv.BandType.BANDPASS, 50)

# We can view the components we created like this.
model.print_components()

# And again we simulate
fs = hv.frange(2e9, 3e9, 1_00_001)

S = model.run_sp(fs)

hv.plot_s_parameters(fs, [S.S11, S.S21], labels=['S11', 'S21'])
