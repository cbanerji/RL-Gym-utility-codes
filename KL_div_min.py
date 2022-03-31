import pandas as pd
import numpy as np
import tensorflow as tf
from numpy import random
import matplotlib.pyplot as plt
np.random.seed(3)

# Define the Loss function (KL divergence for Multivariate Gaussian)-------------
def kl_mvn(m_0,s_0,m_1,s_1):
    # store inv diag covariance of S1 and diff between means
    N_0 = m_0.get_shape().as_list()
    N = N_0[0]
    iS1 = tf.linalg.inv(s_1)
    diff = m_1 - m_0

    # kl div. of Multivariate gaussian consists of three terms
    tr_term   = tf.trace(iS1 @ s_0)
    det_term  = tf.log(tf.linalg.det(s_1)/tf.linalg.det(s_0))
    quad_term = tf.transpose(diff) @ tf.linalg.inv(s_1) @ diff
    return .5 * (tr_term + det_term + quad_term - N)

def dist_create():
    '''
    Generates two distributions for experimentation
    '''
    #--Distribution 'P'
    M0 = np.random.uniform(-1,1,4).reshape(4,1) # Mean vector
    #create a Gaussian diagonal covariance matrix
    S0 = np.zeros((4, 4), float)
    b0 = np.random.uniform(0,1,4)
    np.fill_diagonal(S0, b0)
    #--Distribution 'Q'
    M1 = np.random.uniform(-1,1,4).reshape(4,1) #Mean vector
    S1 = np.zeros((4, 4), float)
    b1 = np.random.uniform(0,1,4)
    np.fill_diagonal(S1, b1)
    return M0, S0, M1, S1

def KL_PQ():
    M0, S0, M1, S1 = dist_create()
    M_0 = tf.Variable(M0, shape = [M0.shape[0],1], dtype=tf.float64) # Convert to tensorflow variable
    S_0 = tf.Variable(S0, dtype=tf.float64)
    M_1 = tf.Variable(tf.constant(M1, shape = [M1.shape[0],1], dtype=tf.float64))
    S_1 = tf.Variable(tf.constant(S1, shape = S1.shape, dtype = tf.float64))
    return M_0,S_0,M_1,S_1

def KL_QP():
    M0, S0, M1, S1 = dist_create()
    M_0 = tf.Variable(M1, shape = [M1.shape[0],1], dtype=tf.float64)
    S_0 = tf.Variable(S1, dtype=tf.float64)
    M_1 = tf.Variable(tf.constant(M0, shape = [M0.shape[0],1], dtype=tf.float64))
    S_1 = tf.Variable(tf.constant(S0, shape = S0.shape, dtype = tf.float64))
    return M_0,S_0,M_1,S_1

# A function to switch between KL(P||Q) and KL(Q||P)-----------------------------------------------
def KL_helper(choice):
        switcher={
                0: KL_PQ,
                1: KL_QP
                }
        func = switcher.get(choice)
        return func()


if __name__ == "__main__":
    pc = 90 # set stopping criteria parameter
    # Get distributions
    M_0,S_0,M_1,S_1 = KL_helper(0) # choose '0' for KL(P|Q) and '1' for KL(Q|P)-----

    #-Set optimizer type and the optimized objective----
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    objective = kl_mvn(M_0,S_0,M_1,S_1) # The KL divergence calculation

    train = optimizer.minimize(objective)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    ful_sav=[]
    sav = []
    means =[]
    sigma=[]

    for iter in range (10000):
        vul = sess.run([train, objective])
        ful_sav.append(vul[1])
        fix = np.array(ful_sav)
        if iter % 10 == 0: # save only after each 100th iteration
            sav.append(vul[1])
            #record shift of gaussian parameters due to divergence minimization
            means.append(sess.run(M_1))
            sigma.append(sess.run(S_1))
            print('Objective='+str(vul[1]))
            print('Vul'+str(vul))

            # Early stopping: Optimization stop at percentage(pc)
            if fix[iter] <= fix[0]-((pc/100)*fix[0]):
                break
