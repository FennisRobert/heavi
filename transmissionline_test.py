import heavi as hv
import numpy as np

M = hv.Model()

n1 = M.port(50)
n2 = M.port(50)

M.TL(n1,n2,lambda f: 2*np.pi*f/3e8, 0.2, 50)

f = hv.frange(1e9, 10e9, 1001)

S = M.run_sparameter_analysis(f)

hv.plot_s_parameters(f, [S.S11, S.S21], labels=["S11", "S21"], linestyles=["-","-"], colorcycle=[0,1])