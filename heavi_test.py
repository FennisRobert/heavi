import heavi as hf
from heavi.lib import smd
import numpy as np
## A sipmle test case of a two port with a 4th order filter and sparameter analaysis and plotting

model = hf.Model()

mc = hf.MonteCarlo()

n1 = model.node()
p1 = model.terminal(n1, 50)
n2 = model.node()
#n3 = model.node()

Z0s = mc.gaussian(50,5)
p2 = model.terminal(n2, Z0s)

#resistor = smd.SMDResistor(5, smd.SMDResistorSize.R0402).connect(n2,n3)

model.filters.cauer_filter(model.gnd, n1, n2, 2e9, 70e6, 5, 0.03, hf.FilterType.CHEBYCHEV, type=hf.BandType.BANDPASS)

f = hf.frange(1.8e9, 2.2e9, 2001)

sweep = hf.ParameterSweep()
sweep.add_dimension(Z0s)

for indices in mc.iterate(10):
    S = model.run_sparameter_analysis(f)
    hf.plot_s_parameters(f, [S.S11, S.S21], labels=["S11", "S21"], linestyles=["-","-"], colorcycle=[0,1])

model.print_components()

# for component in model.components:
#     print(component)
#     for func in component.functionlist:
#         print(f'Value at 1GHz: {func.simval(1e9)}')