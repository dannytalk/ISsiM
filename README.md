# ISsiM

<img src="./docs/ISsiMlogo.png" width="200" height="200">

ISsiM (InterStellar Medium Simulator) is a python package for simulating the gas within galaxies. It is a Smooth Particle Hydrodynamics simulator mixed with a direct object N-body simulator. This strong combination allows you to quickly run a galaxy simulation without advanced knowledge in simulations or other programming languages. 

### Installation

Write this command into your terminal or notebook 

`pip install issim`

The code is also available at <https://pypi.org/project/issim/>

### Example 

```python 
import issim
import numpy as np

# Simulation parameters
N         = 5000    # Number of particles
tEnd      = 3000     # time at which simulation ends
dt        = 1   # timestep
M         = 1      # galaxy mass
R         = 1   # galaxy radius
h         = 0.04    # smoothing length
k         = 0.000004    # equation of state constant
n         = 3      # polytropic index
nu        = 0.000003     # damping
G         = 0.00019160287    #gravitational acceleration
scaleheight =  0.05 #scale height
randvel = 0.0004 #random velocity 
plotRealTime = True # switch on for plotting as the simulation goes along

np.random.seed(42)     
gal1 = issim.galgen.Galaxy(N, mass = M, radius = R, scaleheight = scaleheight,
randvel = randvel,G = G,  k = k , n = n, nu = nu)
simulation = gal1.simulate(smoothinglength = h, tstep = dt, tEnd = tEnd)

animation = gal1.quadplotvid(downsampling = 4)
gal1.saveanim()
```

### Documentation
Documentation is available at <https://issim.readthedocs.io/en/latest/index.html>
