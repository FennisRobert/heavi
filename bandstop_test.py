import heavi as hv

f = hv.frange(1.8e9, 2.3e9, 2001)

# Model with two cascaded filters
model = hv.Model(suppress_loadbar=True)

p1 = model.port(50)
p2 = model.port(50)

n1 = model.node()
n2 = model.node()

model.filters.cauer_filter(model.gnd, p1, n1, 2e9, 100e6, 5, 0.05, 
                           hv.FilterType.CHEBYCHEV, 
                           type=hv.BandType.BANDPASS, 
                           chebychev_correction=True)
model.filters.cauer_filter(model.gnd, n2, p2, 2.1e9, 100e6, 5, 0.05, 
                           hv.FilterType.CHEBYCHEV, 
                           type=hv.BandType.BANDSTOP, 
                           chebychev_correction=True)
model.transmissionline(model.gnd, n1,n2, 51, 2.1*(1-0.5j), 0.02)

S = model.run_sparameter_analysis(f)

# Model with one filter
model2 = hv.Model(suppress_loadbar=True)

p12 = model2.port(50)
p22 = model2.port(50)

model2.filters.cauer_filter(model2.gnd, p12, p22, 2.1e9, 100e6, 5, 0.05, 
                            hv.FilterType.CHEBYCHEV, 
                            type=hv.BandType.BANDSTOP, 
                            chebychev_correction=True)

S2 = model2.run_sparameter_analysis(f)

# Model with one filter
model3 = hv.Model(suppress_loadbar=True)

p13 = model3.port(50)
p23 = model3.port(50)

model3.filters.cauer_filter(model3.gnd, p13, p23, 2.0e9, 100e6, 5, 0.05, 
                            hv.FilterType.CHEBYCHEV, 
                            type=hv.BandType.BANDPASS, 
                            chebychev_correction=True)

S3 = model3.run_sparameter_analysis(f)

# Plotting the data
hv.plot_s_parameters(f, [S.S11, S.S21, S2.S11, S3.S11, S2.S21, S3.S21], 
                     labels=["Dual S11", "Dual S21", "Single S11", "Single S11","Single SS1", "Single SS1"], 
                     linestyles=["-",":","--","--",":",":"], 
                     colorcycle=[0,0,1,2,1,2],
                     dblim=[-40,5])