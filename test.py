from scipy import *
import numpy as np
import animation
import core
import matplotlib.pyplot as plt

eta_vec = np.linspace(0, 5.0, 11)
eta_vec = [0.1]
o_vec = np.zeros_like(eta_vec)

for i, eta in enumerate(eta_vec):
        Bird = core.Birds(L=5, N=300, eta=eta, r=1, dt=1, v=0.1, frame=100, ep=1, mode='competition')
        pos_mat3, vel_mat, norm_vel_vec = Bird.run()

        o_vec[i] = np.mean(norm_vel_vec[-20:])



anim = animation.Simulation_Animation(pos_mat3[:,:,0].T, pos_mat3[:,:,1].T, vel_mat.T, Bird)
