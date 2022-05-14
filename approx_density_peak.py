import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import random
import gym
import numpy as np
import random
import math
random.manual_seed=24

# Generate toy data for experimentation
# Generate 3 gaussians and sample data from them
mean1 = [0.5, 0.5]
cov1 = [[0.01,0],[0,0.01]]
mean2 = [0.9, 0.89]
cov2 = [[0.01,0],[0,0.01]]
mean3 = [0.01, 0.21]
cov3 = [[0.05,0],[0,0.01]]
#print(np.random.multivariate_normal(mean, cov, 2))
samp1 = np.random.multivariate_normal(mean1,cov1,1000)
samp2 = np.random.multivariate_normal(mean2,cov2,800)
samp3 = np.random.multivariate_normal(mean3,cov3,10)
samplr = np.vstack((samp1,samp2,samp3))
plt.plot(samp1[:,0],samp1[:,1],'ro', markersize='2')
plt.plot(samp2[:,0],samp2[:,1],'bo', markersize='2')
plt.plot(samp3[:,0],samp3[:,1],'go', markersize='2')


# Calculate density
def get_dense(can_pk,ng,k):
    '''
    Returns density score of a given Candidates
    can_pk = a candidate density peak
    ng = random sample, to be used for density calculation
    k = sample size (here = 10% of data)
    '''
    iter_dist = []
    for pck in ng:
        # Calculate Euclidean distance between two points
        euc = math.sqrt((can_pk[0] - pck[0])**2 + (can_pk[1] - pck[1])**2)
        #get weight
        wt = np.exp(-euc) # we weight the values according to their distance from the candidate
        iter_dist.append(wt*euc)
    den_score = math.exp(-np.sum(iter_dist)/k) # calculate Density score
    print('**Den score: '+str(den_score))
    return den_score

def get_den_all(candt_pk, data_bufr, k,rept=3):
    '''
    Returns best density peak from given list of candidate density peaks
    '''
    score_arr = np.zeros([len(candt_pk), rept])
    for rp in range (rept):
        ng = random.sample(data_bufr, k) #get k sized random sampled voters from buffer
        print('**')
        for can_id, can_pk in enumerate(candt_pk):
            #Calculate density scores
            #2d array, can_id: score_rept1, score_rept2, score_rept3
            print('can_id %s, can_pk %s, rept %s'%(can_id, can_pk, rp))
            score_arr[can_id][rp] = get_dense(can_pk,ng,k) # get density value for this candidate
            print('Score_array_filled:'+str(score_arr))
    #take row-wise average, i.e. average density value for each canditate
    print('Score_array_filled:'+str(score_arr))
    can_avg_den = np.average(score_arr,axis=1)
    print('The score matrix:'+str(can_avg_den))
    #return candidate peak with maximum density value
    max_indx = np.where(can_avg_den == max(can_avg_den))
    max_indx = np.squeeze(max_indx)
    max_indx = max_indx.item()
    den_peak = candt_pk[max_indx]
    return den_peak

def get_density_pk(data_bufr):
    coll = []
    euc_list = []
    candt = 5 # No. of candidate density peaks
    k = int(0.010 *len(data_bufr)) # 10% data considered for density calculation
    # Sample 'candt' points at random
    candt_pk = random.sample(data_bufr, candt)
    den_pk = get_den_all(candt_pk,data_bufr, k)
    print("\n chosen candidate density peak: "+str(den_pk))
    return den_pk

cen = get_density_pk(samplr.tolist())
plt.plot(cen[0],cen[1], 'k', marker='*', markersize=10)
plt.show()
