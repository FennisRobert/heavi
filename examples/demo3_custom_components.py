"""
Sometimes you want to create custom components that have all sorts of internal behavior. There is a class
intended exactly for that!

In this example we will create the effective component of a resistor with parasitics.

"""

import heavi as hv

from heavi.lib.libgen import SubCircuit
import numpy as np

# In case you want to reate an N-port we will use the SubCircuit class.

class MyResistor(SubCircuit):
   
    def __init__(self, resistance: float):
        super().__init__()
        self.n_nodes = 2 # Here we set it manually We must set this property for the class to work.
        self.resistance: float = resistance
       
    """ To make our component do something we have to use the __on_connect__ dunder method"""
    def __on_connect__(self):

        # There are multiple ways to use your own components. In this case we will use a custom impedance function.
        # We will compute the impedance of a resistor with paracitic capacitance and inductance.
        
        # The values of the parasitics are sort of arbitrary for now
        Lps = 10e-9
        Cpp = 0.5e-12
        
        # The function must compute the series impedance
        # as a function of frequency (in hertz)
        def z_parasitic(f):
            # Z_RLC = R + j*2*pi*f*L + 1/(j*2*pi*f*C)
            w = 2 * 3.141592653589793 * f
            Zr = self.resistance
            Zl = 1j * w * Lps
            Zc = 1 / (1j * w * Cpp)
            # return Zr + Zl parallel to Zc
            Z = (Zr+Zl) * Zc / (Zr + Zl + Zc)
            print(Z)
            return Z
        
        # To put our component in the circuit we can do multiple things.
        # The Network object is available as all Node objects carry a reference to it and this method is called after you connected it to nodes.
        # Thus we can use the network to just plug in all components we want. In this case we can just create an impedance component with a function as value.
        self.function = z_parasitic
        
        # our two nodes that this device is connected two are in self.node(1) and self.node(2)
        # Thus if we did
        # MyResistor.connect(n1,n2)
        # Then self.node(1) would return n1 and self.node(2) would return n2 (in order)
        
        # We add the impedance manually with the z_parasitic function for our resistor. We can set the display value to show what the device is.
        # 
        self.network.impedance(self.node(1), self.node(2), z_parasitic, display_value=self.resistance)\
        .set_metadata(name='Custom Component',
                      unit='Ω',
                      value=self.resistance,
                      inductance=Lps,
                      capacitance=Cpp)
        # This metadata can be used to provide more info on what this device is.
        
# Now we can use it!

m = hv.Model()

n1 = m.quick_port(50)
n2 = m.quick_port(50)

# We can create our resistor and connect it to our nodes. We make a simple 50, 50 ohm resistor devider. Its just for the idea of it.
R1 = MyResistor(50)
R1.connect(n1,n2)

R1 = MyResistor(50)
R1.connect(n2,m.gnd)

# The Frequency range
fs = np.logspace(6,10.3,3001)

S = m.run_sp(fs)

hv.plot_s_parameters(fs, [S.S11,S.S21], labels=['S11','S21'], logx=True, dblim=[-20,0])