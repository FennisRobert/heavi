import heavi as hv

model = hv.Model(suppress_loadbar=True)

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

f = hv.frange(1e6, 10e9,10_001)

Ss = []
for index in sweep.iterate():
    S = model.run_sparameter_analysis(f)
    Ss.append(S.S21)

hv.plot_s_parameters(f, Ss)

