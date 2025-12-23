"""
Parameter Sweeps
In this demo we will quickly discuss how to do Paramter sweeps in heavi. It has a unique system to simplify the process.

"""

# We import Heavi
import heavi as hv

M = hv.Model()

# To execute a parameter sweep we creat the ParameterSweep class. 
sweep = hv.ParameterSweep()

# We can create new sweep axes using the lin method for example.
# First we call .lin(start,stop) to specify a value range. Then we call .add(N) to say: Create N points from start to stop.
Cs = sweep.lin(1e-12,2e-12).add(20)

# Sometimes we want to create two parameters that oth will change simultaneously instead of only one. Thus creating pairs. We can do this by repeatedly calling .lin()
# The .add(N) method will then return each linked parameters.
Rs, Ls = sweep.lin(50,100).lin(1e-9,5e-9).add(10)


# Cs, Rs and Ls are of type Param which is a sort of value that doesn't exist yet.

# Lets create our nodes and define our simple network.
nin = M.quick_port(50)
nout = M.quick_port(50)
nmid = M.node()

# The network is quite random but as you see we plug in the Param objects into our resistor. In this case our Resistor is now not defined by a single value
# but by a parameter that may have its value changed.
M.resistor(nin,nmid,Rs)
M.inductor(nmid,nout,Ls)
M.capacitor(M.gnd,nmid,Cs)

# To simmualte we define our frequency range
f = hv.frange(1e9,2e9,1001)

# We want to iterate over our entire parameter space so we call the iterate() method of our sweep function.
# i will be the nth iteration
for i in sweep.iterate():
    # We can compute the S-parameters and plug them into our sweep object together with the frequency.
    Sparameters = M.run_sp(f)
    sweep.submit(Sparameters,f)

# Finally we can call the MdimSparam function. This will create a multi-dimensional array of in this case shape 20,10,nF due to the order in which we 
# added our sweep axes.

MdimSparam = sweep.generate_mdim()

# We plot all our parameters superimposed. Its not interesting to look at but it serves its purpose.
hv.plot_s_parameters(f, [MdimSparam.S21[i,j,:] for i in range(20) for j in range(10)])