
# RL-Gym-utility-codes
## About:
This repo contains codes relevant to reinforcement learning projects involving Open-AI gym environments.

 + **State dim. redcn.** Performs dimensional reduction on Gyn env. observation space vectors in real time.
  It first trains an autoencoder from data collected by running a random policy in the environment.

+ **KL_div_min.** Minimizes the KL divergence between two multivariate Gaussian distributions. Using Tensorflow, the code minimizes the KL div. objective over iterations.Two randomly generated multivariate Gaussians are considered for experimentation.

+ **approx_density_peak.** Approximates max data density peak from a given dataset.
