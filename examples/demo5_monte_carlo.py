
""" Monte Carlo Simulations

In this example we will show you how to do Monte-Carlo simulations.
"""
import heavi as hv
import numpy as np

M = hv.Model()

# Just like with Parameter sweeps we create a MonteCarlo object which will manage our parameter variations.

mc = hv.MonteCarlo()

# We will create a Capacitor value that is distributed by a Gaussian function.

Cvalue = mc.gaussian(1e-12, 1e-13) # Gaussian distribution with mean=1pF, standard deviation=0.1pF

# Next we create our network with a ristor and capacitor
n1 = M.quick_port(50)
n2 = M.quick_port(50)

M.capacitor(n1,n2,Cvalue)
M.resistor(M.gnd,n2,10_000)

# And we simulate our model
f = hv.frange(1e6,3e9,1001)

# We use the iterate funtion to run our model for 1000 steps
Ss = []
for itt in mc.iterate(201):
    S = M.run_sp(f)
    Ss.append(S)

# And we do some post processing to plot the range of values we may expect.
Ss = np.array([s.S21 for s in Ss])

Smin = np.min(Ss, axis=0)
Smax = np.max(Ss, axis=0)
Smean = np.mean(Ss, axis=0)

hv.plot_s_parameters(f, [Smin, Smean, Smax], labels=['S21 min','S21','S21 max'], linestyles=['--','-','--'], colorcycle=[0,0,0])