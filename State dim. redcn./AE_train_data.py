'''
Trains an Autoencoder using states explored by a random policy in Gym environment.
The trained Encoder will return a low dimensional representation of the higher dimesional state(or observation in Gym) vector.
'''

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import gym
from numpy import random
#from mpl_toolkits import mplot3d

def create_train_data(seed, env, qty= 50000):
    torch.manual_seed(seed)

    #-------------------------Generate exploration data-------------
    print("\n Collecting traning data... \n ")
    env = gym.make(env)
    observation = env.reset()
    tstep, t, obs_col = 0, 0, []
    while tstep < qty:
        t += 1
        tstep +=1
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        obs_col.append(observation)
        print(tstep)
        if done:
            #print("Episode finished after {} timesteps".format(t))
            observation = env.reset()
            t = 0
    exp_dat = np.asarray(obs_col)# collection of all explored states/ observations

    #np.savetxt("state_action_exp_dat_InvPend.csv", exp_dat, delimiter=",") #Save the explored data
    return exp_dat, exp_dat.shape
