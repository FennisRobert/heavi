"""
Optimization

In this example we will quickly show how to optimize your components in Heavi.

We are going to optimize a simple LC circuit to have the right values and cutoff frequency.
"""

import heavi as hv
from heavi.lib import optim

M = hv.Model()

p1 = M.quick_port(50)
p2 = M.quick_port(50)

# To optimize we create an optimizer object.
opt = optim.Optimiser(M)

# We create a new parameter but we want to distribute the range logarithmically.
# Thus we will optimize in the exponent range going from 10pH to 1uH etc. Then we apply a value mapping that computes the actual value.
Lopt = opt.add_param(-9, (-11,-6), mapping=lambda x: 10**x)
Copt = opt.add_param(-12,(-14,-9), mapping=lambda x: 10**x)

# We create our circuit.
M.inductor(p1,p2, Lopt,)
M.capacitor(M.gnd, p2, Copt)

# We are going to define some S-parameter constraint using the add_splimit function. The arguments are:
#  fmin: start frequency
#  fmax: ending frequency
#  nF: Number of frequency points to sample
#  Sp index: The Sparameter to study, in this case firs S11 and then S21
#  upper_limit: This is a funtion that should penalize certain value excurions. We can create a penalty term simply by using the optim.dBbelow function
#  weight: How important it is

# The optim.dBbelow(-15,norm=4) will create a penalty function that gets higher if the value is above the provided limit in dB's. The norm indicates the
# Lebesque norm: Thus, for each frequency f we have: Penalty = (sum penalty**n)**(1/n)

opt.add_splimit(0.1e9, 2e9, 21, (1,1), upper_limit=-15, norm=4, weight=5)
opt.add_splimit(10e9, 20e9, 21, (2,1), upper_limit=-15, norm=4)

# Heavi does not have a build in optimizer but we can simply use ones from Scipy! Differential evolution si very good
from scipy.optimize import differential_evolution

# We call the differential evolution on our objective function. We create this using the generate_objective() method
# Differential weighting is a feature to prevent optimizers from getting stuck at local minima by adding the differene between optimization variables as penalty term

objecte_function = opt.generate_objective(2,differential_weighting=True)

# Finally we can call the differential evolution function. The bounds for each parameter are simply available in the bounds property of our opt instance.
x0 = differential_evolution(objecte_function, bounds=opt.bounds, popsize=50)

print(x0)
M.print_components()

# After the final iteration has been called, that version of the parameters of our circuit are still online! W


fs = hv.frange(1e9,30e9,1001)

S = M.run_sp(fs)

hv.plot_s_parameters(fs,[S.S11,S.S21], labels=['S11','S21'], spec_area=opt.spec_area, logx=True)