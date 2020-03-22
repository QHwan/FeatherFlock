from math import acos, asin
import numpy as np
from numpy.random import rand, uniform
from tqdm import tqdm

class Core(object):
    # class builder initiation 
    def __init__(self, L=25, N=300, eta=.1, r=1, dt=1, frame=50, *args, **kwargs):
        self.L = L #length of LxL domain
        self.N = N #number of particles
        self.eta = eta #noise for direction
        self.r = r #interaction radius
        self.dt = dt #timestep
        self.frame = frame #total iterations
        
    # Set up initial condition
    def set_initial_condition(self):
        pos_mat3 = np.zeros((self.frame, self.N, 2))
        vel_mat = np.zeros((self.frame, self.N))

        pos_mat3[0,:,0] = self.L*rand(self.N) #initial positions x-coordinates
        pos_mat3[0,:,1] = self.L*rand(self.N) #initial poisitions y-coordinates
        vel_mat[0] = 2*np.pi*rand(self.N) #initial velocities
        return(pos_mat3, vel_mat)
    
    def theta_noise(self):
        #This is a uniform disribution of noise
        return(uniform(-self.eta/2,self.eta/2,self.N))
    

class Birds(Core):
    def __init__(self,v=0.03, ep=1, *args,**kwargs):

        self.v = v #constant speed
        self.ep = ep #the delay parameter (ep = 1 implies the Vicsek model, ep = 0 is ballistic movement)
        
        self.args = args
        self.kwargs = kwargs
        
        super(Birds,self).__init__(*args,**kwargs)
    

    def run(self):
        #This function simulates the scheme selected by the user for the flocking mechanism
        
        #Inputs -- None
        #Outputs -- Positions and velocities of all timesteps of the simulation
        pos_mat3, vel_mat = self.set_initial_condition()
        for i in tqdm(range(1, self.frame)):
            pos_mat3[i], vel_mat[i] = self.update_metric(pos_mat3[i-1], vel_mat[i-1])

        return(pos_mat3, vel_mat)
    

    def update_metric(self,pos_mat, vel_vec):
        #This method allows us to update the flocking model with the Metric model
        #Each new velocity is constructed by averaging over all of the velocities within
        #the radius selected, self.r.
        #Inputs -- x-coordinates, y-coordinates, trajectories for time = t
        #Outputs -- x-coordinates, y-coordinates, trajectories for time = t + (delta t)
        avgs = 0*vel_vec
        avgc = 0*vel_vec
        
        for j in range(0,self.N):
            #find distances for all particles
            dist_vec = self.calc_dist(pos_mat, self.L, j)
            
            #find indicies that are within the radius
            ngbs_idx_vec = [i for i in range(len(dist_vec)) if dist_vec[i] <= self.r]
            
            #find average velocity of those inside the radius
            sint = 0 
            cost = 0
            for k in ngbs_idx_vec:
                sint = sint + np.sin(vel_vec[k])
                cost = cost + np.cos(vel_vec[k])
            if (len(ngbs_idx_vec)==0):
                #catch yourself if there are no objects within the desired radius
                avgs[j] = 0
                avgc[j] = 0
            else:
                avgs[j] = sint/len(ngbs_idx_vec)
                avgc[j] = cost/len(ngbs_idx_vec)
                
        #construct the noise
        noise = self.theta_noise()
        
        #update velocities and positions
        cosi = (self.ep)*avgc+(1-self.ep)*np.cos(vel_vec)
        sini = (self.ep)*avgs+(1-self.ep)*np.sin(vel_vec)
        newvelo = np.arctan2(sini,cosi) 
        velon = np.mod(newvelo + noise,2*np.pi)
        pos_mat[:,0] = pos_mat[:,0] + self.dt*self.v*np.cos(velon) 
        pos_mat[:,1] = pos_mat[:,1] + self.dt*self.v*np.sin(velon)
        
        #Make sure that the positions are not outside the boundary.
        #If so, correct for periodicity
        pos_mat = self.check_boundary(pos_mat)
        
        #Outputs returned
        return(pos_mat, velon)
       
    
    def calc_dist(self, pos_mat, L, j):
        #find distance of every particle from particle j using periodic boundary conditions
        posx = pos_mat[:,0]
        posy = pos_mat[:,1]
        Dist0 = np.sqrt((posx[j] - posx)**2 + (posy[j] - posy)**2) #regular  
        Dist1 = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy + L)**2) #topleft
        Dist2 = np.sqrt((posx[j]  - posx)**2 + (posy[j] - posy + L)**2) #topcenter
        Dist3 = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy + L)**2) #topright
        Dist4 = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy)**2) #middleleft
        Dist5 = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy)**2) #middleright
        Dist6 = np.sqrt((posx[j]  - L - posx)**2 + (posy[j] - posy - L)**2) #bottomleft
        Dist7 = np.sqrt((posx[j]  - posx)**2 + (posy[j] - posy - L)**2) #bottomcenter
        Dist8 = np.sqrt((posx[j]  + L - posx)**2 + (posy[j] - posy - L)**2) #bottomright
        
        TD = [Dist0,Dist1,Dist2,Dist3,Dist4,Dist5,Dist6,Dist7,Dist8]
        
        return(np.asarray(TD).min(0)) #minimum values for all possible distances
    
    def check_boundary(self,pos_mat):
        posx = pos_mat[:,0]
        posy = pos_mat[:,1]
        xcordn = [i for i in range(self.N) if posx[i] < 0]
        xcordp = [i for i in range(self.N) if posx[i] > self.L]
        ycordn = [i for i in range(self.N) if posy[i] < 0]
        ycordp = [i for i in range(self.N) if posy[i] > self.L]
        
        for j in xcordn:
            posx[j] = posx[j] + self.L
       
        for j in xcordp:
            posx[j] = posx[j] - self.L
            
        for j in ycordn:
            posy[j] = posy[j] + self.L
            
        for j in ycordp:
            posy[j] = posy[j] - self.L
           
        return(pos_mat)
                                      
