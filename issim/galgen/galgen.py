import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
import scipy.special as special
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display, clear_output
from tqdm import tqdm
from ..issim import simulate

#you need ffmpeg package installed

plt.rcParams['animation.embed_limit'] = 2**128


TrueG = (6.67430 * (10**(-11)) * u.m**3 / (u.kg * u.second**2)) 
SolG = TrueG.to(u.AU**3/(u.solMass * u.year**2))
milkmass = u.def_unit('milkmass', 1.15*(10**12) * u.solMass)
milkrad =  u.def_unit('milkrad', 30 * u.kiloparsec)
MilkG = TrueG.to(milkrad**3/(milkmass * u.megayear**2))

def timepull(simulation, time):
    """
    Find the index of the closest time in a simulation to a given time.

    Parameters:
    - simulation: A tuple containing positions, colors, and times of the simulation.
    - time: The time to find in the simulation.

    Returns:
    - index: The index of the closest time in the simulation.
    """
    times = simulation[2]
    difference_array = np.absolute(times-time)
    # find the index of minimum element from the array
    index = difference_array.argmin()
    return index 

def generate_3d_exponential_cylindrical(n, radius, height):
    """
    Generate 3D coordinates in cylindrical space where the radial distance follows
    a radial exponential distribution and z-coordinates are uniformly distributed.

    Parameters:
    - n: Number of points to generate.
    - lambda_r: Rate parameter (λ) for the radial distance's exponential distribution.
    - z_min: Minimum z-coordinate.
    - z_max: Maximum z-coordinate.

    Returns:
    - A numpy array of shape (n, 3) containing the 3D coordinates.

    
    Example:
    
    >>> import numpy as np
    >>> from issim.galgen import generate_3d_exponential_cylindrical
    
    >>> # Generating 1000 points with a radius of 2 units and a height of 5 units
    >>> points = generate_3d_exponential_cylindrical(1000, 2, 5)
    
    >>> # Check the shape of the generated points
    >>> points.shape
    (1000, 3)
    
    >>> # Visualize the generated points in 3D space
    >>> import matplotlib.pyplot as plt
    >>> from mpl_toolkits.mplot3d import Axes3D
    
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111, projection='3d')
    >>> ax.scatter(points[:,0], points[:,1], points[:,2])
    >>> ax.set_xlabel('X')
    >>> ax.set_ylabel('Y')
    >>> ax.set_zlabel('Z')
    >>> plt.show()
    

    """
    # Radial distances
    r = np.random.exponential(radius, n)
    
    # Azimuthal angle, theta, uniformly distributed between 0 and 2pi
    theta = np.random.uniform(0, 2 * np.pi, n)
    
    # z-coordinates uniformly distributed
    z = np.random.uniform(-height/2, height/2, n)
    
    # Convert cylindrical coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.vstack((x, y, z)).T



def exponentialdiskvelocity(G, M, Rdisk, R):
    '''
    Calculates the angular velocity of an object at a certain radial distance 
    within an exponential disk galaxy using a simplified model. This function 
    assumes an exponential disk mass distribution and utilizes specific Bessel 
    functions to estimate the velocity.

    Parameters:
    - G (float): Gravitational constant. Ensure consistent units are used
                 throughout all parameters.
    - M (float): Total mass of the galaxy or the mass enclosed within a certain
                 radius. Again, consistent units with G are essential.
    - Rdisk (float): Scale length of the galaxy's exponential disk. This parameter
                     defines how rapidly the disk's density decreases with radius
                     from the center of the galaxy.
    - R (float or array_like): Radial distance(s) from the center of the galaxy 
                               at which to calculate the angular velocity. Can be
                               a single value or an array of values for calculating
                               velocities at multiple radii simultaneously.

    Returns:
    - vels (float or ndarray): The angular velocity (or velocities if R is an array) 
                               at the specified radial distance(s) from the galaxy's 
                               center. The return type matches the input type of R.

    Notes:
    - This function uses the special.iv (modified Bessel function of the first kind)
      and special.kv (modified Bessel function of the second kind) from the SciPy
      library to compute the velocities. Ensure SciPy is installed and imported
      correctly in your environment.

    Example:
    
    >>> import numpy as np
    >>> from scipy import special
    >>> G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    >>> M = 1e11  # Mass of the galaxy in kg
    >>> Rdisk = 3.5e3  # Scale length of the galaxy's disk in light-years
    >>> R = np.array([8000, 15000])  # Radial distances in light-years
    >>> velocities = exponentialdiskvelocity(G, M, Rdisk, R)
    >>> print(velocities)
    

    The output will be the angular velocities at the specified radial distances.
    '''
    yvals = R/(2*Rdisk)
    frontvalus = ((2*G*M)/Rdisk) * yvals**2
    bessels = (special.iv(0,yvals) * special.kv(0,yvals)) - (special.iv(1,yvals) * special.kv(1,yvals))
    vels = np.sqrt(frontvalus*bessels)
    return vels

def normalize_vectors(vectors):
    """
    Normalize a list or array of 3D vectors to their unit vectors.
    
    Parameters:
    - vectors: A list or a numpy array of shape (n, 3) where n is the number of vectors
    
    Returns:
    - A numpy array of shape (n, 3) containing the unit vectors.

    Example:
    >>> import numpy as np
    >>> from issim.galgen import normalize_vectors
    
    >>> # Define some 3D vectors
    >>> vectors = np.array([[1, 2, 2], [3, 4, 5], [6, 7, 8]])
    
    >>> # Normalize the vectors
    >>> normalized_vectors = normalize_vectors(vectors)
    
    >>> # Check the normalized vectors
    >>> normalized_vectors
    array([[0.33333333, 0.66666667, 0.66666667],
           [0.42426407, 0.56568542, 0.70710678],
           [0.46666667, 0.54772256, 0.6289709 ]])
    
    >>> # Check if the magnitudes of the normalized vectors are approximately 1
    >>> np.linalg.norm(normalized_vectors, axis=1)
    array([1., 1., 1.])
    
    """
    vectors = np.array(vectors)  # Ensure input is converted to a numpy array
    magnitudes = np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Calculate magnitudes and reshape for broadcasting
    unit_vectors = vectors / magnitudes  # Normalize
    
    return unit_vectors

def radiuscalc2Dcylinder(vectors):
    """
    Calculate the radial positions of 3D vectors projected onto a 2D cylindrical space.

    Parameters:
    - vectors: A numpy array of shape (n, 3) containing the 3D vectors.

    Returns:
    - A numpy array of shape (n,) containing the radial positions of the vectors in 2D cylindrical space.
    
    The radial position of a vector in cylindrical coordinates is calculated as the Euclidean distance 
    from the origin to the projection of the vector onto the xy-plane.

    Example:
    >>> import numpy as np
    >>> from issim.galgen import radiuscalc2Dcylinder
    
    >>> # Define some 3D vectors
    >>> vectors = np.array([[1, 2, 2], [3, 4, 5], [6, 7, 8]])
    
    >>> # Calculate the radial positions of the vectors in 2D cylindrical space
    >>> radial_positions = radiuscalc2Dcylinder(vectors)
    
    >>> # Check the radial positions
    >>> radial_positions
    array([2.23606798, 5.        , 9.21954446])
    """

    radvectors = np.sqrt( (vectors[:,0])**2 + (vectors[:,1])**2)
    return radvectors 


class Galaxy():
    '''
    A class for generating the initial conditions for a spiral galaxy, running simulations, and generating animations of those simulations. 
    '''
    def __init__(self, N, mass, radius, scaleheight, randvel,G =0.00019160287,  k = 0.4246 , n = 4/3, nu = 0.1):
        """
        Initialize a Galaxy object.

        Parameters:
        - N: The number of particles in the galaxy.
        - mass: The mass of the galaxy.
        - radius: The radius of the galaxy.
        - scaleheight: The scale height of the galaxy.
        - randvel: The magnitude of random velocities for the particles.
        - G: The gravitational constant (default is 0.00019160287).
        - k: The equation of state constant (default is 0.4246).
        - n: The polytropic index (default is 4/3).
        - nu: The viscosity coefficient (default is 0.1).
        """
        
        #stored constants 
        self.particles = N
        self.mass = mass    
        self.radius = radius
        self.scaleheight = scaleheight
        self.randomvel = randvel
        self.Grav = G
        self.konstant = k
        self.pindex = n
        self.viscosity = nu
        self.simulation = None 
        #generate the galaxy 
        print('...generating galaxy')
        self.particlemass = self.mass/self.particles * np.ones(self.particles)
        self.positions = generate_3d_exponential_cylindrical(self.particles, self.radius, self.scaleheight)  
        self.radialvelocities =  exponentialdiskvelocity( self.Grav,self.mass,self.radius, radiuscalc2Dcylinder(self.positions) )
        self.randomvelocities = self.randomvel * np.random.randn(self.particles, 3)
        self.velocities = (normalize_vectors(np.cross(self.positions, [0,0,1])) * (self.radialvelocities)[:, np.newaxis]) + self.randomvelocities
        print('...galaxy generated')
        
    def regenerate(self):
        """
        Regenerate the galaxy with the same parameters.
        """
        #generate the galaxy 
        print('...generating galaxy')
        self.particlemass = self.mass/self.particles * np.ones(self.particles)
        self.positions = generate_3d_exponential_cylindrical(self.particles, self.radius, self.scaleheight)   
        self.radialvelocities =  exponentialdiskvelocity( self.Grav,self.mass,self.radius, radiuscalc2Dcylinder(self.positions) )
        self.randomvelocities = self.randomvel * np.random.randn(self.particles, 3)
        self.velocities = (normalize_vectors(np.cross(self.positions, [0,0,1])) * (self.radialvelocities)[:, np.newaxis]) + self.randomvelocities
        print('...galaxy generated')

    def simulate(self, smoothinglength, tstep, tEnd, tStart = 0):
        """
        Simulate the movement of particles in the galaxy.

        Parameters:
        - smoothinglength: The smoothing length for SPH calculations.
        - tstep: The time step for integration.
        - tEnd: The end time of the simulation.
        - tStart: The start time of the simulation (default is 0).

        Returns:
        - simulation: A tuple containing positions, colors, and times of the simulation.
        """
        self.simulation = simulate( N = self.particles, tStart = tStart, tstep = tstep, tEnd = tEnd, pos = self.positions, vel = self.velocities, masses = self.particlemass, 
                                    h = smoothinglength, k =  self.konstant, n = self.pindex, nu = self.viscosity, G = self.Grav, simradius = self.radius, plotRealTime = True) 
        return self.simulation
    def plot(self, time, orientation = 'top', xlim = None, ylim =None, cmap = plt.cm.autumn, alpha=0.25,  s=10, **kwargs):
        """
        Plot the galaxy at a specific time.

        Parameters:
        - time: The time at which to plot the galaxy.
        - orientation: The orientation of the plot ('top' or 'side', default is 'top').
        - xlim: The x-axis limits of the plot (default is None).
        - ylim: The y-axis limits of the plot (default is None).
        - cmap: The colormap for density plotting (default is plt.cm.autumn).
        - alpha: The transparency of points (default is 0.25).
        - s: The size of points (default is 10).
        """

        if self.simulation is None:
            print('NO SIMULATION HAS BEEN RUN YET')
            return

        if xlim is None:
            xlim = (-4.5*self.radius, 4.5*self.radius)
        if ylim is None:
            ylim = (-4.5*self.radius, 4.5*self.radius)
            
        posarr = self.simulation[0]
        colarr = self.simulation[1]
        t = timepull(self.simulation, time)
        print(f'Graphed at time {self.simulation[2][t]}')

        fig, ax = plt.subplots()
        ax.set_facecolor('black')
        ax.set_facecolor((.1,.1,.1))
        if orientation == 'top':
            plt.scatter(posarr[t][:,0], posarr[t][:,1], c=(colarr[t]), cmap=cmap, alpha = alpha , s = s, **kwargs )
        elif orientation == 'side':
            plt.scatter(posarr[t][:,0], posarr[t][:,2], c=(colarr[t]), cmap=cmap, alpha = alpha , s = s, **kwargs )
        else:
            print('NOT A VALID VIEWING DIRECTION')
        plt.xlim(xlim)
        plt.ylim(ylim)
    def quadplot(self, time, xlim1 = None, ylim1 =None, xlim2 =None, ylim2 = None,  xlim3 = None, ylim3 = None, 
                  xlim4 = None, ylim4 = None, cmap = plt.cm.autumn, alpha=0.25,  s=10, **kwargs):
        """
        Plot the galaxy in four different orientations at a specific time.

        Parameters:
        - time: The time at which to plot the galaxy.
        - xlim1, ylim1: The x-axis and y-axis limits for the first subplot (default is None).
        - xlim2, ylim2: The x-axis and y-axis limits for the second subplot (default is None).
        - xlim3, ylim3: The x-axis and y-axis limits for the third subplot (default is None).
        - xlim4, ylim4: The x-axis and y-axis limits for the fourth subplot (default is None).
        - cmap: The colormap for density plotting (default is plt.cm.autumn).
        - alpha: The transparency of points (default is 0.25).
        - s: The size of points (default is 10).
        """
        
        if self.simulation is None:
            print('NO SIMULATION HAS BEEN RUN YET')
            return

        if xlim1 is None:
            xlim1 = (-4.5*self.radius, 4.5*self.radius)
        if ylim1 is None:
            ylim1 = (-4.5*self.radius, 4.5*self.radius)
        if xlim2 is None:
            xlim2 = (-4.5*self.radius, 4.5*self.radius)
        if ylim2 is None:
            ylim2 = (-4.5*self.radius, 4.5*self.radius)
        if xlim3 is None:
            xlim3 = (-self.radius, self.radius)
        if ylim3 is None:
            ylim3 = (-self.radius, self.radius)
        if xlim4 is None:
            xlim4 = (-10*self.radius, 10*self.radius)
        if ylim4 is None:
            ylim4 = (-10*self.radius, 10*self.radius)

        
        posarr = self.simulation[0]
        colarr = self.simulation[1]
        t = timepull(self.simulation, time)
        print(f'Graphed at time {self.simulation[2][t]}')
        fig = plt.figure(figsize=(10,12), dpi=100)
        grid = plt.GridSpec(2, 2, wspace=0.0, hspace=0.3)
        ax1 = plt.subplot(grid[0,0])
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,0])
        ax4 = plt.subplot(grid[1,1])
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
    
        ax1.set(xlim=xlim1, ylim=ylim1)
        ax1.set_aspect('equal', 'box')
        ax1.set_xticks([-1,0,1])
        ax1.set_yticks([-1,0,1])
        ax1.set_facecolor('black')
        ax1.set_facecolor((.1,.1,.1))
        ax1.scatter(posarr[t][:,0], posarr[t][:,1], c=(colarr[t]), cmap=cmap, s=s, alpha=alpha, **kwargs)
    
        ax2.set(xlim=xlim2, ylim=ylim2)
        ax2.set_aspect('equal', 'box')
        ax2.set_xticks([-1,0,1])
        ax2.set_yticks([-1,0,1])
        ax2.set_facecolor('black')
        ax2.set_facecolor((.1,.1,.1)) 
        ax2.scatter(posarr[t][:,0], posarr[t][:,2], c=(colarr[t]), cmap=cmap, s=s, alpha=alpha, **kwargs)

        ax3.set(xlim=xlim3, ylim=ylim3)
        ax3.set_aspect('equal', 'box')
        ax3.set_xticks([-1,0,1])
        ax3.set_yticks([-1,0,1])
        ax3.set_facecolor('black')
        ax3.set_facecolor((.1,.1,.1))
        ax3.scatter(posarr[t][:,0], posarr[t][:,1], c=(colarr[t]), cmap=cmap, s=s, alpha=alpha, **kwargs)
    
        ax4.set(xlim=xlim4, ylim=ylim4)
        ax4.set_aspect('equal', 'box')
        ax4.set_xticks([-1,0,1])
        ax4.set_yticks([-1,0,1])
        ax4.set_facecolor('black')
        ax4.set_facecolor((.1,.1,.1))
        ax4.scatter(posarr[t][:,0], posarr[t][:,1], c=(colarr[t]), cmap=cmap, s=s, alpha=alpha, **kwargs)
    def plotvid(self, downsampling=1, orientation = 'top', xlim = None, ylim =None, cmap = plt.cm.autumn, alpha=0.25,  s=10, **kwargs):
        """
        Create a video of the galaxy's movement.

        Parameters:
        - downsampling: The downsampling factor for the video (default is 1).
        - orientation: The orientation of the plot ('top' or 'side', default is 'top').
        - xlim: The x-axis limits of the plot (default is None).
        - ylim: The y-axis limits of the plot (default is None).
        - cmap: The colormap for density plotting (default is plt.cm.autumn).
        - alpha: The transparency of points (default is 0.25).
        - s: The size of points (default is 10).
        """
        if self.simulation is None:
            print('NO SIMULATION HAS BEEN RUN YET')
            return
        if xlim is None:
            xlim = (-4.5*self.radius, 4.5*self.radius)
        if ylim is None:
            ylim = (-4.5*self.radius, 4.5*self.radius)
        print('unfinished')
    def quadplotvid(self, downsampling=1, xlim1 =None, ylim1 =None, xlim2 = None, ylim2 =None,  xlim3 = None, 
                    ylim3 = None, xlim4 = None, ylim4 = None, cmap = plt.cm.autumn, alpha=0.25,  s=10, **kwargs):
        """
        Create a quad-view video of the galaxy's movement.

        Parameters:
        - downsampling: The downsampling factor for the video (default is 1).
        - orientation: The orientation of the plot ('top' or 'side', default is 'top').
        - xlim1, ylim1: The x-axis and y-axis limits for the first subplot (default is None).
        - xlim2, ylim2: The x-axis and y-axis limits for the second subplot (default is None).
        - xlim3, ylim3: The x-axis and y-axis limits for the third subplot (default is None).
        - xlim4, ylim4: The x-axis and y-axis limits for the fourth subplot (default is None).
        - cmap: The colormap for density plotting (default is plt.cm.autumn).
        - alpha: The transparency of points (default is 0.25).
        - s: The size of points (default is 10).
        """
        if self.simulation is None:
            print('NO SIMULATION HAS BEEN RUN YET')
            return
        
        if xlim1 is None:
            xlim1 = (-4.5*self.radius, 4.5*self.radius)
        if ylim1 is None:
            ylim1 = (-4.5*self.radius, 4.5*self.radius)
        if xlim2 is None:
            xlim2 = (-4.5*self.radius, 4.5*self.radius)
        if ylim2 is None:
            ylim2 = (-4.5*self.radius, 4.5*self.radius)
        if xlim3 is None:
            xlim3 = (-self.radius, self.radius)
        if ylim3 is None:
            ylim3 = (-self.radius, self.radius)
        if xlim4 is None:
            xlim4 = (-10*self.radius, 10*self.radius)
        if ylim4 is None:
            ylim4 = (-10*self.radius, 10*self.radius)
        
        
        # Downsampling by selecting every nth point
        posarr = self.simulation[0]
        colarr = self.simulation[1]
        
        positions1 = posarr[::downsampling]
        colors1 = colarr[::downsampling] 

    
        def update_plot(i):
            ax1.cla()
            ax2.cla()
            ax3.cla()
            ax4.cla()
            
    
            ax1.set(xlim=xlim1, ylim=ylim1)
            ax1.set_aspect('equal', 'box')
            ax1.set_xticks([-1,0,1])
            ax1.set_yticks([-1,0,1])
            ax1.set_facecolor('black')
            ax1.set_facecolor((.1,.1,.1))
            ax1.scatter(positions1[i][:,0], positions1[i][:,1], c=(colors1[i]),  cmap=cmap, s=s, alpha=alpha, **kwargs)
    
            ax2.set(xlim=xlim2, ylim=ylim2)
            ax2.set_aspect('equal', 'box')
            ax2.set_xticks([-1,0,1])
            ax2.set_yticks([-1,0,1])
            ax2.set_facecolor('black')
            ax2.set_facecolor((.1,.1,.1)) 
            ax2.scatter(positions1[i][:,0], positions1[i][:,2], c=(colors1[i]), cmap=cmap, s=s, alpha=alpha, **kwargs)

            ax3.set(xlim=xlim3, ylim=ylim3)
            ax3.set_aspect('equal', 'box')
            ax3.set_xticks([-1,0,1])
            ax3.set_yticks([-1,0,1])
            ax3.set_facecolor('black')
            ax3.set_facecolor((.1,.1,.1))
            ax3.scatter(positions1[i][:,0], positions1[i][:,1], c=(colors1[i]),  cmap=cmap, s=s, alpha=alpha, **kwargs)
    
            ax4.set(xlim=xlim4, ylim=ylim4)
            ax4.set_aspect('equal', 'box')
            ax4.set_xticks([-1,0,1])
            ax4.set_yticks([-1,0,1])
            ax4.set_facecolor('black')
            ax4.set_facecolor((.1,.1,.1))
            ax4.scatter(positions1[i][:,0], positions1[i][:,1], c=(colors1[i]),  cmap=cmap, s=s, alpha=alpha, **kwargs)

        fig = plt.figure(figsize=(10,12), dpi=100)
        grid = plt.GridSpec(2, 2, wspace=0.0, hspace=0.3)
        ax1 = plt.subplot(grid[0,0])
        ax2 = plt.subplot(grid[0,1])
        ax3 = plt.subplot(grid[1,0])
        ax4 = plt.subplot(grid[1,1])

        #positions, colors = main()
        ani = FuncAnimation(fig, update_plot, frames=len(positions1), interval=50, repeat=False)
        
        display(HTML(ani.to_jshtml()))
        
        return ani


            
            
        
        


            
            
            
        
        
