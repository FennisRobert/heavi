import heavi as hv

model = hv.Model()

p1 = model.port(50)
p2 = model.port(50)

n1 = model.node()

sweep = hv.ParameterSweep()
R1, R2 = sweep.lin(50,100).lin(20,200).add(5)

C3 = hv.Param.lin(1e-12, 10e-12, 10)

sweep.add_dimension(C3)

R1 = model.resistor(p1, n1, R1)

R2 = model.resistor(n1, p2, R2)

C = model.capacitor(n1, model.gnd, C3)

f = hv.frange(1e6, 10e9,1001)

for index in sweep.iterate():
    S = model.run_sparameter_analysis(f)
    hv.plot_s_parameters(f, [S.S11, S.S21], labels=["S11", "S21"], linestyles=["-","-"], colorcycle=[0,1])

