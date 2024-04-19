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
Create Your Own Smoothed-Particle-Hydrodynamics Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate the structure of a star with SPH
"""

def W( x, y, z, h ):
	"""
    Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	w     is the evaluated smoothing function
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	w = (1.0 / (h*np.sqrt(np.pi)))**3 * np.exp( -r**2 / h**2)
	
	return w
	
	
def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
	
	
def getPairwiseSeparations( ri, rj ):
	"""
	Get pairwise desprations between 2 sets of coordinates
	ri    is an M x 3 matrix of positions
	rj    is an N x 3 matrix of positions
	dx, dy, dz   are M x N matrices of separations
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
    Get Density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """
    M = r.shape[0]
	
    dx, dy, dz = getPairwiseSeparations( r, pos );

    rho = np.sum( m * W(dx, dy, dz, h), 1 ).reshape((M,1)) #CHECK IF THIS MULTIPLICATION IS CORRECT

    return rho
	
	
def getPressure(rho, k, n):
	"""
	Equation of State
	rho   vector of densities
	k     equation of state constant
	n     polytropic index
	P     pressure
	"""
	
	P = k * rho**(1+1/n)
	
	return P

def arbacalc3d_vectorized(G, positions, masses):
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

def radpressure(G, positions, masses):
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
    Calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     equation of state constant
    n     polytropic index
    nu    viscosity
    a     is N x 3 matrix of accelerations
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