import heavi as hv
from heavi.lib import tl
from heavi.lib import lumped

M = hv.Model()

Z2 = 200

p1 = M.new_port(50)
p2 = M.new_port(Z2)


p1 > lumped.TL(50, 0.05) > M(1)
M(1) > tl.Transformer(50, Z2, 2e9, 3, 0.01) > M(2)
M(2) > lumped.TL(Z2, 0.05) > p2

fs = hv.frange(0.01e9,4e9,10_000)

S = M.run(fs)

M.print_components()

hv.plot_s_parameters(fs,[S.S11,S.S21], labels=['S11', 'S21'])