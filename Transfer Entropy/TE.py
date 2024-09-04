# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:58:44 2024

@author: samue
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_data(real_data, sfreq, noise=1, load_mode=False):
    
    if load_mode:
        sim_data = np.load('./sim_data.npy')
        names = np.load('./sim_data_names.npy')
        names = names.tolist()
        print("Simulated data loaded from file.")
        
        if real_data.shape[0] != sim_data.shape[0]:
            print('Warning: Loaded data contains a different number of channels.')
        if real_data.shape[1] != sim_data.shape[1]:
            print('Warning: Loaded data has a different length')
    else:
        """
        SETUP
        #"semi-random" dataset, each point moves a small amout from the last point,
        # making for more natural 'movement' of data
        """
        channels, samples = real_data.shape[0], real_data.shape[1]
        
        names = []
        for i in range(channels):
            names.append(f'channel_{i}')
        
        data = []
        for i in range(channels):
            r = [np.random.rand()]
            for j in range(samples - 1):
                r.append(0.5 * r[-1] + np.random.rand())
        
            data.append(r)
        data = np.array(data)
        
        ijs = []
        if len(ijs) == 0:
            ijs = []
            N = data.shape[0]
            for i in range(N):
                for j in range(i, N):
                    if not np.all(data[i] == data[j]):
                        ijs.append((i, j))
        
        sim_data = force_causality(data, noise = 1)
        
        np.save('./sim_data.npy', sim_data)
        np.save('./sim_data_names.npy', names)
        print('New simulated data created and saved to file.')

        
    
    return sim_data, names

def force_causality(data, noise=1):

    # amount of noise added, 0.5 is 25% noise

    """
    from 'indepentent' to 'dependent' at an order of 'lag'
    we should see strong granger causality

    method 1
    copies one channel to another, adding some random noise
    rolls channel by a certain lag
    """
    forced_data = data

    independent_channel = 1
    dependent_channel = 5
    lag = 5

    for sample in range(np.shape(data)[1]):
        forced_data[dependent_channel, sample] = data[independent_channel, sample] + (noise * (np.random.rand() - 0.5))

    forced_data[dependent_channel, :] = np.roll(data[dependent_channel, :], lag)

    """
    dataframe = pd.DataFrame(columns=['col1','col2'], data=zip(data[independent_channel,:],data[dependent_channel,:]))
    gctest = grangercausalitytests(dataframe, maxlag=20, verbose=False)
    
    
    method 2
    adds half of independent to half of dependent, which keeps some of the information of the dependent channel
    rolls by a certain lag
    """

    independent_channel = 2
    dependent_channel = 6
    lag = 5

    for sample in range(np.shape(data)[1]):
        forced_data[dependent_channel, sample] = ((0.5 * data[dependent_channel, sample]) +
                                                  (0.5 * data[independent_channel, sample]) +
                                                  (noise * (np.random.rand() - 0.5)))

    forced_data[dependent_channel, :] = np.roll(data[dependent_channel, :], lag)

    """
    Method 3, non-linear
    adding a small quadratic relationship
    and a threshold
    """

    independent_channel = 3
    dependent_channel = 7
    lag1 = 3


    for sample in range(np.shape(data)[1]):
        x = forced_data[dependent_channel, sample]
        y = x**2 if x < 1 else data[independent_channel, sample]

        
        forced_data[independent_channel, sample] = y
    
    """
    MODIFYING CHANNEL 4 TO TEST IF THE CODE FINDS ANY MODIFICATION
    """
    independent_channel = 4
    dependent_channel = 4
    lag1 = 5
    
    for channel in [0,1,2,3,4,8,9]:
        for sample in range(np.shape(data)[1]):
            forced_data[channel, sample] = ((0.5 * data[channel, sample]) +
                                            (0.5 * data[channel, sample]) +
                                            (noise * (np.random.rand() - 0.5)))

    return forced_data

#%%

sfreq = 1024
data = np.zeros([10,sfreq*7])
load = True

sim_data, sim_names = create_data(data, sfreq, noise=5, load_mode=load)


def te(data, k):
    import pyinform
    
    chs = data.shape[0]
    digi_data = np.zeros_like(data)
    
    te = np.zeros([chs,chs,k])
    te_norm = np.zeros_like(te)
    data_avg = np.zeros([1,chs])
    
    print(f'k = ', end = '')
    for k in range(1,k):
        print(k, end=' ')
        for i in range(chs):
            for j in range(chs):
                
                #bins = np.linspace(0, 1, 1)
                #X = np.digitize(data[i],bins)
                #Y = np.digitize(data[j],bins)
                X = data[i]
                Y = data[j]
                #performing te
                te[i,j,k] = pyinform.transferentropy.transfer_entropy(X, Y, k) if i != j else np.nan
                
                data_avg[0,i] = np.average(X)
                digi_data[i] = X
        
        #normalizaing te from 0 to 1        
        te_norm[:,:,k] = (te[:,:,k] - np.nanmin(te[:,:,k])) / (np.nanmax(te[:,:,k]) - np.nanmin(te[:,:,k]))
    
    #taking averages for all history lengths
    te_avg = np.zeros([chs,chs])
    for i in range(chs):
        for j in range(chs):
            te_avg[i,j] = np.average(te_norm[i,j,:])
        
    
    te_result = {"te": te, "te normalized": te_norm, "te average": te_avg, "data average": data_avg, "digitized data": digi_data}
    
    
    return te_result

te = te(sim_data, k=10)

#%%Test
t = [1,2,3,np.nan]
print(np.max(t))
print(np.min(t))

from pyinform import transfer_entropy
te_test = [0]
ks = []
for k in range(1,7):   
    xs = [[1,0,0,0,2,1,1,1,1], [1,1,1,1,0,0,0,1,1]]
    ys = [[0,0,2,1,1,1,0,0,0], [1,0,0,0,0,1,1,1,0]]
    te_test.append(transfer_entropy(xs, ys, k))
    ks.append(k)

tedf = pd.DataFrame(columns=['k      te'], data=te_test)
print(tedf)

#%%


import pyinform

np.random.seed(42)
source_weak = np.random.randint(0, 2, 100)
destination_weak = np.random.randint(0, 2, 100)

# Generate a strong information transfer time series
source_strong = np.random.randint(0, 2, 100)
destination_strong = np.roll(source_strong, 1)  # Destination is a shifted version of source

# Calculate transfer entropy for weak information transfer
te_weak = pyinform.transferentropy.transfer_entropy(source_weak.tolist(), destination_weak.tolist(), k=1)

# Calculate transfer entropy for strong information transfer
te_strong = pyinform.transferentropy.transfer_entropy(source_strong.tolist(), destination_strong.tolist(), k=1)

# Display results
print(f"Transfer Entropy for weak information transfer: {te_weak}. (Discrete 1/0)")
print(f"Transfer Entropy for strong information transfer: {te_strong}. (Discrete)")


# Generate a weak information transfer time series
np.random.seed(42)
source_weak = np.random.rand(100)  # Random values between 0 and 1
destination_weak = np.random.rand(100)  # Independent random values between 0 and 1

# Generate a strong information transfer time series
source_strong = np.random.rand(100)  # Random values between 0 and 1
noise = np.random.normal(0, 0.1, 100)  # Small noise
destination_strong = source_strong + noise  # Destination is source with added noise
destination_strong = np.clip(destination_strong, 0, 1)  # Ensure values stay between 0 and 1

# Calculate transfer entropy for weak information transfer
te_weak = pyinform.transferentropy.transfer_entropy(source_weak.tolist(), destination_weak.tolist(), k=1)

# Calculate transfer entropy for strong information transfer
te_strong = pyinform.transferentropy.transfer_entropy(source_strong.tolist(), destination_strong.tolist(), k=1)

# Display results
print(f"Transfer Entropy for weak information transfer: {te_weak} (Continuous)")
print(f"Transfer Entropy for strong information transfer: {te_strong} (Continuous)")

#%%

import numpy as np
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors


# Function to compute Transfer Entropy (TE) using nearest neighbors
def compute_te(x, y, k=4, u=1):
    n_samples = len(x)
    
    # Delay embedding
    X = np.array([x[i:n_samples - u + i] for i in range(u)]).T
    Y = np.array([y[i:n_samples - u + i] for i in range(u)]).T
    Y_next = y[u:]
    
    # Nearest neighbor search for probabilities
    nbrs_X = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    nbrs_XY = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(np.hstack((X, Y)))
    
    # Distances to k-th nearest neighbor
    dist_X, _ = nbrs_X.kneighbors(X)
    dist_XY, _ = nbrs_XY.kneighbors(np.hstack((X, Y)))
    
    # Estimate the TE using Kullback-Leibler divergence
    te = np.mean(np.log(dist_X[:, -1] / dist_XY[:, -1]))
    
    return te

# Function to run permutation tests to assess significance
def permutation_test(x, y, num_permutations=1000, k=4, u=1):
    observed_te = compute_te(x, y, k=k, u=u)
    permuted_te_values = []
    
    for _ in range(num_permutations):
        y_permuted = np.random.permutation(y)
        permuted_te = compute_te(x, y_permuted, k=k, u=u)
        permuted_te_values.append(permuted_te)
    
    p_value = np.sum(np.array(permuted_te_values) >= observed_te) / num_permutations
    return observed_te, p_value

x1 = sim_data[3,:]  # Channel 1
y5 = sim_data[7,:]  # Channel 5
y2 = sim_data[4]  # Channel 2

# Compute Transfer Entropy for Channel 1 -> Channel 5
te_value_1_to_5, p_value_1_to_5 = permutation_test(x1, y5)

print(f"Transfer Entropy from Channel 1 to Channel 5: {te_value_1_to_5:.4f}")
print(f"P-value (1 -> 5): {p_value_1_to_5:.4f}")

# Compute Transfer Entropy for Channel 1 -> Channel 2
te_value_1_to_2, p_value_1_to_2 = permutation_test(x1, y2)

print(f"Transfer Entropy from Channel 1 to Channel 2: {te_value_1_to_2:.4f}")
print(f"P-value (1 -> 2): {p_value_1_to_2:.4f}")

# Comparison of TE values
print("\nComparison:")
if p_value_1_to_5 < 0.05:
    print("Significant causality detected from Channel 1 to Channel 5.")
else:
    print("No significant causality detected from Channel 1 to Channel 5.")

if p_value_1_to_2 < 0.05:
    print("Significant causality detected from Channel 1 to Channel 2.")
else:
    print("No significant causality detected from Channel 1 to Channel 2.")

