import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import random
from mpl_toolkits import mplot3d
import AE_train_data as dim
import AEncoder as tae


if __name__=="__main__":
    '''
    env: the OpenAI gym environment # ID
    seed: for replicable results, manually setting random seed
    epoch: no. of training epochs for the autoencoder.
    '''
    env = "InvertedPendulum-v2"
    seed = 17
    epoch = 10
    dat, dat_shape = dim.create_train_data(seed, env) #Gather training data
    mod = tae.train_AE(dat, dat_shape, epoch) #Train autoencoder, the trained model is returned.
