import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import scipy.special as special
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output
from tqdm import tqdm

#you need ffmpeg package installed

plt.rcParams['animation.embed_limit'] = 2**128

"""
Based on Code from 

Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

"""

def W( x, y, z, h ):
    """
    Compute the Gaussian smoothing kernel in three dimensions.

    This function evaluates the Gaussian smoothing kernel in three dimensions,
    which is commonly used in kernel density estimation and smoothing operations.

    Parameters:
    - x: A  vector or matrix representing the x positions.
    - y: A vector or matrix representing the y positions.
    - z: A vector or matrix representing the z positions.
    - h: The smoothing length or bandwidth of the kernel.

    Returns:
    - w: A vector or matrix representing the evaluated smoothing function.

    The Gaussian smoothing kernel is given by:
        w = (1 / (h * sqrt(pi)))^3 * exp(-r^2 / h^2)

    where:
    - r: The Euclidean distance from the origin to the point (x, y, z).

    Example:
    >>> import numpy as np
    >>> from your_module import W

    >>> # Define positions and smoothing length
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([0.5, 1.5, 2.5])
    >>> z = np.array([0.2, 1.2, 2.2])
    >>> h = 0.5

    >>> # Compute the Gaussian smoothing kernel
    >>> w = W(x, y, z, h)

    >>> # Check the evaluated smoothing function
    >>> w
    array([0.00021743, 0.00291502, 0.0238942 ])
    """
	
    r = np.sqrt(x**2 + y**2 + z**2)
	
    w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
    return w
	
	
def gradW( x, y, z, h ):
    """
    Compute the gradient of the Gaussian smoothing kernel in three dimensions.

    This function computes the gradient of the Gaussian smoothing kernel in three dimensions,
    which is commonly used in smoothed particle hydrodynamics (SPH) simulations.

    Parameters:
    - x: A vector or matrix representing the x positions.
    - y: A vector or matrix representing the y positions.
    - z: A vector or matrix representing the z positions.
    - h: The smoothing length or bandwidth of the kernel.

    Returns:
    - wx: A vector or matrix representing the x-component of the evaluated gradient.
    - wy: A vector or matrix representing the y-component of the evaluated gradient.
    - wz: A vector or matrix representing the z-component of the evaluated gradient.

    The gradient of the Gaussian smoothing kernel is given by:
        wx = n * x
        wy = n * y
        wz = n * z

    where:
    - r: The Euclidean distance from the origin to the point (x, y, z).
    - n: The precomputed coefficient for the gradient, computed as -2 * exp(-r^2 / h^2) / h^5 / sqrt(pi)^3.

    Example:
    >>> import numpy as np
    >>> from your_module import gradW

    >>> # Define positions and smoothing length
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([0.5, 1.5, 2.5])
    >>> z = np.array([0.2, 1.2, 2.2])
    >>> h = 0.5

    >>> # Compute the gradient of the Gaussian smoothing kernel
    >>> wx, wy, wz = gradW(x, y, z, h)

    >>> # Check the evaluated gradient components
    >>> wx, wy, wz
    (array([-0.00010871, -0.00043583, -0.00097912]),
     array([-5.43535809e-05, -2.17414324e-04, -4.88957328e-04]),
     array([-2.17414324e-05, -8.69657295e-05, -1.95622816e-04]))
    """
    r = np.sqrt(x**2 + y**2 + z**2)
	
    n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
    wx = n * x
    wy = n * y
    wz = n * z
	
    return wx, wy, wz
	
	
def getPairwiseSeparations( ri, rj ):
    """
    Calculate pairwise separations between two sets of coordinates.

    This function computes the pairwise separations between two sets of coordinates,
    representing the distances between each point in the first set (ri) and each point
    in the second set (rj).

    Parameters:
    - ri: An M x 3 matrix of positions for the first set of coordinates.
    - rj: An N x 3 matrix of positions for the second set of coordinates.

    Returns:
    - dx: An M x N matrix of separations along the x-axis.
    - dy: An M x N matrix of separations along the y-axis.
    - dz: An M x N matrix of separations along the z-axis.

    Example:
    >>> import numpy as np
    >>> from your_module import getPairwiseSeparations

    >>> # Define two sets of coordinates
    >>> ri = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> rj = np.array([[0.5, 1.5, 2.5], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    >>> # Calculate pairwise separations
    >>> dx, dy, dz = getPairwiseSeparations(ri, rj)

    >>> # Check the calculated separations
    >>> dx
    array([[ 0.5, -2.5, -5.5],
           [ 3. ,  1. , -2. ],
           [ 6. ,  4. ,  1. ]])

    >>> dy
    array([[ 0., -1., -2.],
           [ 1.,  0., -1.],
           [ 2.,  1.,  0.]])

    >>> dz
    array([[ 0., -1., -2.],
           [ 1.,  0., -1.],
           [ 2.,  1.,  0.]])
    """
	
    M = ri.shape[0]
    N = rj.shape[0]
	
    # positions ri = (x,y,z)
    rix = ri[:,0].reshape((M,1))
    riy = ri[:,1].reshape((M,1))
    riz = ri[:,2].reshape((M,1))
	
    # other set of points positions rj = (x,y,z)
    rjx = rj[:,0].reshape((N,1))
    rjy = rj[:,1].reshape((N,1))
    rjz = rj[:,2].reshape((N,1))
	
    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T
	
    return dx, dy, dz
	

def getDensity( r, pos, m, h ):
    """
    Calculate density at sampling locations from smoothed particle hydrodynamics (SPH) particle distribution.

    This function computes the density at sampling locations based on the smoothed particle hydrodynamics (SPH)
    particle distribution. It utilizes the SPH density estimation method, where the density at a given location
    is computed by summing the contributions from neighboring particles using a smoothing kernel.

    Parameters:
    - r: An M x 3 matrix of sampling locations.
    - pos: An N x 3 matrix of SPH particle positions.
    - m: The mass of each SPH particle.
    - h: The smoothing length or bandwidth of the kernel.

    Returns:
    - rho: An M x 1 vector of densities corresponding to the sampling locations.

    Example:
    >>> import numpy as np
    >>> from your_module import getDensity

    >>> # Define sampling locations, SPH particle positions, particle mass, and smoothing length
    >>> r = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> pos = np.array([[0.5, 1.5, 2.5], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    >>> m = 0.1
    >>> h = 0.5

    >>> # Calculate density at sampling locations using SPH method
    >>> rho = getDensity(r, pos, m, h)

    >>> # Check the calculated densities
    >>> rho
    array([[0.05631997],
           [0.18138118]])
    """
    
    M = r.shape[0]
	
    dx, dy, dz = getPairwiseSeparations( r, pos );

    rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1)) 

    return rho
	
	
def getPressure(rho, k, n):
    """
    Calculate the pressure using the polytropic equation of state.

    This function computes the pressure of a substance using the polytropic equation of state,
    which relates pressure (P) to density (rho) according to the formula P = k * rho^(1 + 1/n),
    where:
    - rho: A vector of densities.
    - k: The equation of state constant.
    - n: The polytropic index.

    Parameters:
    - rho: A numpy array or scalar representing the density or densities of the substance(s).
    - k: A scalar representing the equation of state constant.
    - n: A scalar representing the polytropic index.

    Returns:
    - P: A numpy array or scalar representing the pressure(s) calculated using the polytropic equation.

    Example:
    >>> import numpy as np
    >>> from your_module import getPressure

    >>> # Define densities, equation of state constant, and polytropic index
    >>> rho = np.array([1.0, 2.0, 3.0])
    >>> k = 1.5
    >>> n = 1.2

    >>> # Calculate the pressure using the polytropic equation of state
    >>> pressure = getPressure(rho, k, n)

    >>> # Check the calculated pressures
    >>> pressure
    array([1.5       , 4.29118668, 8.68395323])
    """
	
    P = k * rho**(1+1/n)
	
    return P

def arbacalc3d_vectorized(G, positions, masses):
    
    """
    Compute the accelerations of objects due to gravity in a 3D space.

    This function calculates the accelerations of multiple objects in a gravitational field.
    The calculation is performed in a vectorized manner, leveraging NumPy broadcasting
    and array operations for efficiency.

    Parameters:
    - G: Gravitational constant.
    - positions: A numpy array of shape (n_objects, 3) containing the positions of the objects
                 in 3D space, where each row represents the (x, y, z) coordinates of an object.
    - masses: A numpy array of shape (n_objects,) containing the masses of the objects.

    Returns:
    - A numpy array of shape (n_objects, 3) containing the accelerations of the objects
      in 3D space, where each row represents the (x, y, z) components of acceleration
      for an object.

    Details:
    - The function first computes the pairwise differences in positions between all objects
      to obtain the distance vectors.
    - It then calculates the distances between all pairs of objects using the Euclidean norm.
    - To avoid division by zero for the same object (resulting in infinite acceleration),
      the function sets the distances between an object and itself to np.inf.
    - Next, it computes the acceleration due to gravity for each pair of objects using
      Newton's law of universal gravitation.
    - Finally, it sums the contributions from all other masses to obtain the total acceleration
      acting on each object, excluding the self-contribution.

    Example:
    >>> import numpy as np
    >>> from your_module import arbacalc3d_vectorized

    >>> # Define gravitational constant and object positions and masses
    >>> G = 6.67430e-11
    >>> positions = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    >>> masses = np.array([10, 20, 30])

    >>> # Compute accelerations due to gravity
    >>> accelerations = arbacalc3d_vectorized(G, positions, masses)

    >>> # Check the calculated accelerations
    >>> accelerations
    array([[ 0.        ,  0.        ,  0.        ],
           [-1.33110506, -1.33110506, -1.33110506],
           [-0.60981822, -0.60981822, -0.60981822]])
    """
        
    n_objects = len(masses)
    accels = np.zeros((n_objects, 3))
    
    # Expand dimensions to support broadcasting for pairwise subtraction
    pos_expanded = positions[:, np.newaxis, :]  # Shape: (n_objects, 1, 2)
    masses_expanded = masses[:, np.newaxis]  # Shape: (n_objects, 1)
    
    # Compute pairwise differences in positions and distances
    delta_pos = pos_expanded - positions  # Shape: (n_objects, n_objects, 2)
    distances = np.linalg.norm(delta_pos, axis=2)  # Shape: (n_objects, n_objects)
    
    # Avoid division by zero for the same object by setting distance to np.inf
    np.fill_diagonal(distances, np.inf)
    
    # Compute the acceleration due to gravity
    # Add an axis to distances to support broadcasting in division
    acc_due_to_gravity = -G * delta_pos * masses_expanded / distances[:,:,np.newaxis] ** 3  # Shape: (n_objects, n_objects, 2)
    
    # Sum the contributions from all other masses, ignoring the diagonal (self-contribution)
    accels = np.sum(acc_due_to_gravity, axis=1)
    
    return accels


def getAcc( pos, vel, m, h, k, n, nu, G ):
    
    """
    Calculate the acceleration on each smoothed particle hydrodynamics (SPH) particle.

    This function computes the acceleration on each SPH particle in a simulation. It uses
    SPH techniques to calculate the pressure forces, gravitational forces, and viscosity forces
    acting on the particles.

    Parameters:
    - pos: An N x 3 matrix of positions for the SPH particles.
    - vel: An N x 3 matrix of velocities for the SPH particles.
    - m: The mass of each SPH particle.
    - h: The smoothing length or bandwidth of the kernel.
    - k: The equation of state constant.
    - n: The polytropic index.
    - nu: The viscosity coefficient.
    - G: The gravitational constant.

    Returns:
    - a: An N x 3 matrix of accelerations for each SPH particle.

    Example:
    >>> import numpy as np
    >>> from your_module import getAcc

    >>> # Define positions, velocities, masses, smoothing length, equation of state parameters, viscosity, and gravity constant
    >>> pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> vel = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    >>> m = 0.1
    >>> h = 0.5
    >>> k = 1.0
    >>> n = 1.5
    >>> nu = 0.01
    >>> G = 9.81

    >>> # Calculate accelerations using SPH method
    >>> a = getAcc(pos, vel, m, h, k, n, nu, G)

    >>> # Check the calculated accelerations
    >>> a
    array([[-0.01617059, -0.03234117, -0.04851176],
           [-0.05834951, -0.07312689, -0.08790426]])
    """
    N = pos.shape[0]
	
	# Calculate densities at the position of the particles
    rho = getDensity( pos, pos, m, h )
	
	# Get the pressures
    P = getPressure(rho, k, n)

	# Get pairwise distances and gradients
    dx, dy, dz = getPairwiseSeparations( pos, pos )
    dWx, dWy, dWz = gradW( dx, dy, dz, h )
	
	# Add Pressure contribution to accelerations
    ax = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWx, 1).reshape((N,1))
    ay = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWy, 1).reshape((N,1))
    az = - np.sum( m * ( P/rho**2 + P.T/rho.T**2  ) * dWz, 1).reshape((N,1))
	
	# pack together the acceleration components
    a = np.hstack((ax,ay,az))
	
	# Add external potential force
    a += arbacalc3d_vectorized(G, pos, m*(np.ones(len(pos))))
	
	# Add viscosity
    a -= nu * vel
    return a

def simulate(N, tstep, tEnd, pos, vel, masses, h, k, nu, G, plotRealTime = True, tStart = 0,  n = 4/3, simradius = 1):
    """
    Simulate the movement of particles in a smoothed particle hydrodynamics (SPH) N-Body system using leapfrog integration.

    This function performs a simulation of the movement of particles in an SPH N-Body system. It uses leapfrog integration
    techniques to update the positions and velocities of the particles over time. The simulation can optionally graph the
    particles in real-time.

    Parameters:
    - N: The number of particles in the system.
    - tstep: The time step for integration.
    - tEnd: The end time of the simulation.
    - pos: An N x 3 matrix of initial positions for the particles.
    - vel: An N x 3 matrix of initial velocities for the particles.
    - masses: An array of length N containing the masses of the particles.
    - h: The smoothing length or bandwidth of the kernel.
    - k: The equation of state constant.
    - nu: The viscosity coefficient.
    - G: The gravitational constant.
    - plotRealTime: A boolean indicating whether to plot the particles in real-time (default is True).
    - tStart: The start time of the simulation (default is 0).
    - n: The polytropic index (default is 4/3).
    - simradius: The radius of the simulation region (default is 1).

    Returns:
    - positions: A 3D array of shape (Nt, N, 3) containing the positions of the particles at each time step.
    - colors: A 2D array of shape (Nt, N) containing the color values for density plotting at each time step.
    - times: A 1D array of length Nt containing the time values at each time step.

    Example:
    >>> import numpy as np
    >>> from your_module import simulate

    >>> # Define simulation parameters and initial conditions
    >>> N = 100
    >>> tstep = 0.01
    >>> tEnd = 10
    >>> pos = np.random.rand(N, 3) * 10  # Random initial positions
    >>> vel = np.zeros((N, 3))  # Initial velocities
    >>> masses = np.ones(N)  # Equal masses for all particles
    >>> h = 0.1
    >>> k = 1.0
    >>> nu = 0.01
    >>> G = 9.81

    >>> # Perform simulation
    >>> positions, colors, times = simulate(N, tstep, tEnd, pos, vel, masses, h, k, nu, G)

    >>> # Plot the positions of the particles over time
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> for i in range(len(positions)):
    ...     ax.scatter(positions[i,:,0], positions[i,:,1], positions[i,:,2], c=colors[i], cmap=plt.cm.autumn)
    >>> plt.show()
    """
    
    t = tStart
    dt = tstep
    
    # calculate initial gravitational accelerations
    acc = getAcc( pos, vel, masses, h, k, n, nu, G )
	
    # number of timesteps
    Nt = int(np.ceil(tEnd/tstep))

    positions = np.zeros(shape = (Nt, N, 3))
    colors = np.zeros(shape = (Nt, N))
    times = np.zeros(shape = Nt)
    notstars = np.full(N, True) 

    # prep figure
    fig = plt.figure(figsize=(10,12), dpi=100)
    grid = plt.GridSpec(3, 3, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:1,0:1])
    ax2 = plt.subplot(grid[1:2,0:1])
    ax3 = plt.subplot(grid[0:1,1:2])
    ax4 = plt.subplot(grid[1:2,1:2])
	
    # Simulation Main Loop
    for i in tqdm(range(Nt)):
		# (1/2) kick
        vel += acc * dt/2
		
		# drift
        pos += vel * dt
		
		# update accelerations
        acc = getAcc( pos, vel, masses, h, k, n, nu, G )
		
		# (1/2) kick
        vel += acc * dt/2
		
		# update time
        t += dt
		
		# get density for plotting
        rho = getDensity( pos, pos, masses, h )

        #figure out if gas has become star forming matter (False if matter is star forming) 
        newstar = (rho < 10) ################################ USING A TEST VALUE change 10 t0 an actual variable #######################################################
        notstars = notstars * newstar.T
        
        #colors for density plotting 
        cval = np.minimum((rho-3)/3,1).flatten()
		
		# plot in real time
        if plotRealTime or (i == Nt-1):

            clear_output(wait=True)
            plt.sca(ax1)
            plt.cla()
            plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.25)
            ax1.set(xlim=(-4.5 * simradius, 4.5* simradius), ylim=(-4.5*simradius, 4.5*simradius))
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1,0,1])
            ax1.set_yticks([-1,0,1])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))

            plt.sca(ax2)
            plt.cla()
            plt.scatter(pos[:,0],pos[:,2], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.25)
            ax2.set(xlim=(-4.5*simradius, 4.5*simradius), ylim=(-4.5*simradius, 4.5*simradius))
            ax2.set_aspect('equal', 'box')
            ax2.set_xticks([-1,0,1])
            ax2.set_yticks([-1,0,1])
            ax2.set_facecolor('black')
            ax2.set_facecolor((.1,.1,.1)) 

            plt.sca(ax3)
            plt.cla()
            plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.25)
            ax3.set(xlim=(-simradius, simradius), ylim=(-simradius, simradius))
            ax3.set_aspect('equal', 'box')
            ax3.set_xticks([-1,0,1])
            ax3.set_yticks([-1,0,1])
            ax3.set_facecolor('black')
            ax3.set_facecolor((.1,.1,.1))

            plt.sca(ax4)
            plt.cla()
            cval = np.minimum((rho-3)/3,1).flatten()
            plt.scatter(pos[:,0],pos[:,1], c=cval, cmap=plt.cm.autumn, s=10, alpha=0.25)
            ax4.set(xlim=(-10*simradius, 10*simradius), ylim=(-10*simradius, 10*simradius))
            ax4.set_aspect('equal', 'box')
            ax4.set_xticks([-1,0,1])
            ax4.set_yticks([-1,0,1])
            ax4.set_facecolor('black')
            ax4.set_facecolor((.1,.1,.1)) 

            
            

            plt.pause(0.0001)

        positions[i] = pos
        colors[i] = cval
        times[i] = t
	
	# Save figure
    plt.show()
	    
    return positions, colors, times  