from scipy import *
import numpy as np
import animation
import core

Bird = core.Birds(L=5, N=300, eta=.1, r=1, dt=1, v=0.1, frame=100, ep=1)
pos_mat3, vel_mat = Bird.run()


anim = animation.Simulation_Animation(pos_mat3[:,:,0].T, pos_mat3[:,:,1].T, vel_mat.T, Bird)
