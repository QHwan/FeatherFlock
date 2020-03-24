from math import acos, asin
import numpy as np
from numpy.random import rand, uniform
from numpy.core.umath import arctan2, sqrt, sin, cos
from tqdm import tqdm

class Core(object):
    # class builder initiation 
    def __init__(self, L=25, N=300, eta=.1, r=1, dt=1, frame=50, mode='cooperation', *args, **kwargs):
        self.L = L #length of LxL domain
        self.N = N #number of particles
        self.eta = eta #noise for direction
        self.r = r #interaction radius
        self.dt = dt #timestep
        self.frame = frame #total iterations
        self.mode = mode #mode of collective behavior. cooperation and competition
        

    # Set up initial condition
    def set_initial_condition(self):
        self.status_vec = np.zeros(self.N, dtype=int) # status of bird. 0: cooperative, 1: competitive
        pos_mat3 = np.zeros((self.frame, self.N, 2))
        ang_mat = np.zeros((self.frame, self.N))
        norm_vel_vec = np.zeros(self.frame) #normalized absolute velocities

        if self.mode.lower() == 'cooperation':
            pass
        elif self.mode.lower() == 'competition':
            self.status_vec[int(self.N/2):] += 1
        else:
            print("You choose wrong mode.")
            exit(1)
        pos_mat3[0,:,0] = self.L*rand(self.N) #initial positions x-coordinates
        pos_mat3[0,:,1] = self.L*rand(self.N) #initial poisitions y-coordinates
        ang_mat[0] = 2*np.pi*rand(self.N)
        return(pos_mat3, ang_mat, norm_vel_vec)
    

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
        pos_mat3, ang_mat, norm_vel_vec = self.set_initial_condition()
        norm_vel_vec[0] = self.calc_norm_vel(ang_mat[0])
        for i in tqdm(range(1, self.frame)):
            pos_mat3[i], ang_mat[i] = self.update_metric(pos_mat3[i-1], ang_mat[i-1])
            norm_vel_vec[i] = self.calc_norm_vel(ang_mat[i])
        return(pos_mat3, ang_mat, norm_vel_vec)
    
    
    def calc_norm_vel(self, ang_vec):
        vel_sum_vec = np.zeros(2)
        vel_sum_vec[0] += self.v*np.sum(cos(ang_vec))
        vel_sum_vec[1] += self.v*np.sum(sin(ang_vec))
        norm_vel = np.linalg.norm(vel_sum_vec)
        return(norm_vel/self.N/self.v)
            

    def update_metric(self,pos_mat, ang_vec):
        #This method allows us to update the flocking model with the Metric model
        #Each new velocity is constructed by averaging over all of the velocities within
        #the radius selected, self.r.
        #Inputs -- x-coordinates, y-coordinates, trajectories for time = t
        #Outputs -- x-coordinates, y-coordinates, trajectories for time = t + (delta t)
        avg_sin = 0*ang_vec
        avg_cos = 0*ang_vec

        for j in range(0,self.N):
            #find distances for all particles
            dist_vec = self.calc_dist(pos_mat, j)
            
            #find indicies that are within the radius
            ngbs_idx_vec = np.where(dist_vec <= self.r)[0]

            if len(ngbs_idx_vec) == 0:
                avg_sin[j] = 0
                avg_cos[j] = 0
            else:
                #find average velocity of those inside the radius
                sint = np.average(sin(ang_vec[ngbs_idx_vec]))
                cost = np.average(cos(ang_vec[ngbs_idx_vec]))
                avg_sin[j] = sint
                avg_cos[j] = cost
                
        #construct the noise
        noise = self.theta_noise()
        
        #update velocities and positions
        cosi = (self.ep)*avg_cos+(1-self.ep)*np.cos(ang_vec)
        sini = (self.ep)*avg_sin+(1-self.ep)*np.sin(ang_vec)
        new_ang_vec = arctan2(sini,cosi)+noise
        mask = self.status_vec==1
        new_ang_vec[mask] += np.pi
        new_mod_ang_vec = np.mod(new_ang_vec, 2*np.pi)

        pos_mat[:,0] = pos_mat[:,0] + self.dt*self.v*cos(new_mod_ang_vec) 
        pos_mat[:,1] = pos_mat[:,1] + self.dt*self.v*sin(new_mod_ang_vec)
        
        #Make sure that the positions are not outside the boundary.
        #If so, correct for periodicity
        pos_mat = self.check_boundary(pos_mat)
        
        #Outputs returned
        return(pos_mat, new_ang_vec)
       
    
    def calc_dist(self, pos_mat, j):
        #find distance of every particle from particle j using periodic boundary conditions
        ref_pos_vec = pos_mat[j]
        pbc_pos_mat = self.check_pbc(np.tile(ref_pos_vec, (self.N,1)), pos_mat)
    
        dist_vec = np.zeros(self.N)
        dist_vec.fill(self.L*10)
        mask1 = abs(ref_pos_vec[0] - pbc_pos_mat[:,0]) < self.r
        mask2 = abs(ref_pos_vec[1] - pbc_pos_mat[:,1]) < self.r
        mask = mask1 & mask2
        dist_vec[mask] = sqrt((ref_pos_vec[0]-pbc_pos_mat[mask,0])**2 + (ref_pos_vec[1]-pbc_pos_mat[mask,1])**2)
        return(dist_vec)


    def check_pbc(self, ref_pos_mat, pos_mat):    
        pbc_pos_mat = np.copy(pos_mat)
        mask1 = ref_pos_mat - pos_mat > self.L/2
        mask2 = pos_mat - ref_pos_mat > self.L/2
        pbc_pos_mat[mask1] += self.L
        pbc_pos_mat[mask2] -= self.L
        return(pbc_pos_mat)


    def check_boundary(self, pos_mat):
        mask1 = pos_mat < 0
        mask2 = pos_mat > self.L
        pos_mat[mask1] += self.L
        pos_mat[mask2] -= self.L
        return(pos_mat)

                                     
