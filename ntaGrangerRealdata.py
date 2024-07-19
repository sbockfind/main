# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 10:10:42 2024

@author: samue
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import time
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

from tqdm import tqdm

import csv

import nitime.timeseries as ts
import nitime.analysis as nta

import networkx as nx

from numpy import load
import nitime.utils as tsu

from statsmodels.tsa.tsatools import lagmat2ds
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy import stats

def create_data(data, sfreq):
    """
    SETUP
    #"semi-random" dataset, each point moves a small amout from the last point,
    # making for more natural 'movement' of data
    """
    
    channels, samples = data.shape[0], data.shape[1]

    seizure_onset = 2 * sfreq  # seizure is said to happen at 2s in the data
    
    names = []
    for i in range(channels):
        names.append(f'channel {i + 1}')
    
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
    
    return data, names


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
    lag = 2

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

    independent_channel = 1
    dependent_channel = 6
    lag = 2

    for sample in range(np.shape(data)[1]):
        forced_data[dependent_channel, sample] = ((0.5 * data[dependent_channel, sample]) +
                                                  (0.5 * data[independent_channel, sample]) +
                                                  (noise * (np.random.rand() - 0.5)))

    forced_data[dependent_channel, :] = np.roll(data[dependent_channel, :], lag)

    """
    method 2 twice, i.e. double lag
    the independent will cause the dependent at two lags
    also adds noise twice
    
    adds half of independent to half of dependent, which keeps some of the information of the dependent channel
    rolls by a certain lag
    """

    independent_channel = 3
    dependent_channel = 7
    lag1 = 3
    lag2 = 10

    for sample in range(np.shape(data)[1]):
        forced_data[dependent_channel, sample] = ((0.5 * data[dependent_channel, sample]) +
                                                  (0.5 * data[independent_channel, sample]) +
                                                  (noise * (np.random.rand() - 0.5)))

    data[dependent_channel, :] = np.roll(data[dependent_channel, :], lag2-lag1)

    for sample in range(np.shape(data)[1]):
        forced_data[dependent_channel, sample] = ((0.5 * data[dependent_channel, sample]) +
                                                  (0.5 * data[independent_channel, sample]) +
                                                  (noise * (np.random.rand() - 0.5)))

    forced_data[dependent_channel, :] = np.roll(data[dependent_channel, :], lag1)

    return forced_data


def load_electrode_data(data_path, metadata_path):
    data = np.load(data_path)
        #r'C:\Users\samue\OneDrive\FindNeuro\Practice\Granger\ntaGrangerRealdata\electrodes_data_sub10_Seizure3_bipolar.npy')
    data = np.squeeze(data)
    metadata = np.load(metadata_path)
        #r'C:\Users\samue\OneDrive\FindNeuro\Practice\Granger\ntaGrangerRealdata\electrodes_meta_data_sub10_Seizure3_bipolar.npz',
        #allow_pickle=True)
    
    names = metadata['names']
    times = metadata['times']
    sfreq = 1 / (times[1] - times[0])
    
    #seizure_onset = int(2 * sfreq)
    
    #  data_start = seizure_onset - int(0.5 * sfreq)
    #  data_end = seizure_onset + int(0.5 * sfreq)
    #  data = data[:, data_start:data_end]
    
    #  define parameters for the size of the data
    channels, samples = data.shape[0], data.shape[1]
    
    ijs = []
    if len(ijs) == 0:
        ijs = []
        N = data.shape[0]
        for i in range(N):
            for j in range(i, N):
                if not np.all(data[i] == data[j]):
                    ijs.append((i, j))
                    
    return data, names, sfreq, ijs


def crop_channels(data, channel_names):

    def get_prefix(name):    #  to get unique prefixes from channel names
        return name.split('-')[0][:-2]
    
    unique_prefixes = sorted(set(get_prefix(name) for name in channel_names))# Identify unique prefixes
    
    selected_channels = []
    selected_channel_names = []

    for prefix in unique_prefixes:
        # Find all channels with the current prefix
        channels_with_prefix = [i for i, name in enumerate(channel_names) if get_prefix(name) == prefix]

        if len(channels_with_prefix) >= 3:        # Select the first, middle, and last channels
            first = channels_with_prefix[0]
            middle = channels_with_prefix[len(channels_with_prefix) // 2]
            last = channels_with_prefix[-1]
            
            selected_channels.extend([first, middle, last])
            selected_channel_names.extend([channel_names[middle]])
            
    selected_channel_names.extend([channel_names[i] for i in [29,30] if i not in selected_channels])
    
    cropped_data = data[selected_channels, :]
    
    return cropped_data, selected_channel_names


def stationarity_test(data):
    num_of_channels, samples = data[:, :].shape

    maxlag = 20  # define maxlags to test both in stationarity and granger

    # data[126, :3500, 0] = data[126, :3500, 0] / 2  #  force some non-stationarity on last row

    failed_stationarity = []  # empty lists to append later
    stationarity_p_values = []

    for i in tqdm(range(num_of_channels), desc="Processing stationarity"):
        stationarity = adfuller(data[i, :], maxlag=maxlag)
        stationarity_p_values.append(stationarity[1])
        if stationarity[1] > 0.05:
            failed_stationarity.append(i)

    if not failed_stationarity:
        print('Stationarity test passed for all channels.')
    else:
        print('Stationarity test failed on channels:')
        for element in failed_stationarity:
            print(element + 1)

    return stationarity_p_values, failed_stationarity


def lrtest(series_from, series_to, lag=1, verbose=False):


    x = pd.DataFrame(columns=['c1', 'c2'], data=zip(series_to, series_from))

    # create lagmat of both time series
    dta = lagmat2ds(x, lag, trim="both", dropex=1)  # must iterate through all lag

    dtaown = add_constant(dta[:, 1: (lag + 1)], prepend=False)
    dtajoint = add_constant(dta[:, 1:], prepend=False)
    if (
            dtajoint.shape[1] == (dta.shape[1] - 1)
            or (dtajoint.max(0) == dtajoint.min(0)).sum() != 1
    ):
        raise InfeasibleTestError(
            "The x values include a column with constant values and so"
            " the test statistic cannot be computed."
        )

    #  Run ols on both models without and with lags of second variable
    res2down = OLS(dta[:, 0], dtaown).fit()
    res2djoint = OLS(dta[:, 0], dtajoint).fit()

    # likelihood ratio test pvalue:
    lr = -2 * (res2down.llf - res2djoint.llf)
    if verbose:
        print(
            "likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d"
            % (lr, stats.chi2.sf(lr, lag), lag)
        )
    result = (lr, stats.chi2.sf(lr, lag), lag)

    return result


def calc_epoch_granger_causality_ij(time_series, i, j, order, sfreq, fmin=None, fmax=None):

    # Create the time window (t_from->t_to)
    # time_series = ts.TimeSeries(epoch_ts[:, t_from: t_to], sampling_interval=1 / sfreq)
    # Calculate the Granger Causality for all ijs for a specific lag (order)
    # If we want to estimate the order, order is None, and it's being estimated from z to max_order. `
    # Otherwise, max_order is not being used
    # G = nta.GrangerAnalyzer(time_series, order=order, ij=[(i, j)])
    # Create the time window (t_from->t_to)
    ij_chans = time_series[[i, j]]
    time_series = ts.TimeSeries(ij_chans, sampling_interval=1 / sfreq)
    # Calculate the Granger Causality for all ijs for a specific lag (order)
    # If we want to estimate the order, order is None, and it's being estimated from z to max_order. `
    # Otherwise, max_order is not being used
    # WARNING: seems like the ij parameter is not working, so we need to calculate it for all the pairs
    G = nta.GrangerAnalyzer(time_series, order=int(order))  # , ij=[(i, j)]

    freq_idx_G = np.arange(len(G.frequencies))

    gc_val = np.nanmean(G.causality_xy[0, 1, freq_idx_G], -1)
    # gc_val = np.mean(G.causality_xy[i, j, freq_idx_G], -1)
    gc_val = gc_val if not np.isnan(gc_val) else 0
    # [10, 1] is how 10 influences 1. It's in g1 (lower tri).
    # [1, 10] is how 1 influences 10. It's the [10, 1] value in g2, and [1, 10] in g2.T
    # Therefore, the full influence matrix is g1 + g2.T
    # Remember: The upper tri of g1 and g2 are all zeros
    return gc_val


"""
ij = [(i, j) for i in range(channels) for j in range(i + 1, channels + 1)]
nta_likelihoods = ntagranger(cut_data,ijs=ij,order=5,max_order=20,sfreq=sfreq)
"""


def nitime_granger(data, sfreq, channel_names, data_durations=None, maxlag=20):
    """
    performing granger
    """
    num_channels = data.shape[0]  # define size of matrix
    
    if data_durations is None:  # unless specified when the function is called, set the length of data taken to be 2 seconds
        data_durations = [2]
    # creating empty matrices to be filled
    best_likelihood = np.zeros((num_channels, num_channels, max(data_durations) + 1))
    order_best_likelihood = np.zeros((num_channels, num_channels, max(data_durations) + 1))
    nta_gc = np.zeros((num_channels, num_channels, max(data_durations) + 1))
    # main loop
    for duration in data_durations:  # for each duration of data
        # cutting data
        seizure_onset = int(2 * sfreq)  # seizure is at 2 seconds
        data_start = seizure_onset - int(duration / 2 * sfreq)  # 1 second before onset
        data_end = data_start + int(duration * sfreq)  # variable 2,3,4 seconds of data
        cut_data = data[:, data_start:data_end]
        # for each combination of channels
        print(cut_data.shape)
        for i in tqdm(range(num_channels), desc=f"Processing lrtest and gc for {duration} seconds of data"):
            prefix_i = channel_names[i].split('-')[0][:-2]
            for j in range(num_channels):
                prefix_j = channel_names[j].split('-')[0][:-2]
                #if prefix_i == prefix_j:  # skip channels that have the same prefix(they are the same electrode)
                    #print(f'skipped {i}-{j}, same electrode')
                    #continue
                if not np.array_equal(cut_data[i, :], cut_data[j, :]):
                    # using lrtest from grangercausalitytests to find likelihood p-val
                    for lag in range(maxlag):
                        lag = lag+1
                        lr = lrtest(cut_data[i, :], cut_data[j, :], lag)[1]  # from i-row to j-col
                        lr = -np.log10(lr + 0.00000001)
                        # save lowest p-value and, more importantly, the order(lag) that produces that p-val
                        if lr > best_likelihood[i, j, duration]:
                            best_likelihood[i, j, duration] = lr
                            order_best_likelihood[i, j, duration] = lag
                    # find the granger value, if the p-val is significant
                    if best_likelihood[i, j, duration] > -np.log10(0.050000001):
                        #print('doing gc')
                        nta_gc[i, j, duration] = calc_epoch_granger_causality_ij(cut_data, i, j,
                                                                                 order_best_likelihood[i, j, duration],
                                                                                 sfreq)  # from i-row to j-col

    return nta_gc, best_likelihood, order_best_likelihood


def nitime_granger_ij(data, sfreq, i, j, data_duration=None, maxlag=20):
    #single iteration, primarliy for testing
    if data_duration is None:
        data_duration = 2

    seizure_onset = int(2 * sfreq) #  seizure is at 2 seconds
    data_start = seizure_onset - int(data_duration / 2 * sfreq)  #half before, half after
    data_end = data_start + int(data_duration * sfreq)  #  variable 2,3,4 seconds of data
    cut_data = data[:, data_start:data_end]
    order_best_likelihood = 0
    best_likelihood = 0
    nta_gc = 0
    
    if abs(i-j) <= 2: #  skip channels that adjacent by one channel
        print('Adjacent Channels...')    
    elif np.array_equal(cut_data[i, :], cut_data[j, :]):
        print('Channels equal...')
    else:
        
        #  using lrtest from grangercausalitytests to find likelihood p-val
        for lag in range(maxlag):
            lr = -np.log10(
                lrtest(cut_data[i, :], cut_data[j, :], lag)[1] + 0.00000001)  # from i-row to j-col
            #  save lowest p-value and, more importantly, the order(lag) that produces that p-val
            if lr > best_likelihood:
                best_likelihood = lr
                order_best_likelihood = lag
        #  find the granger value, if the p-val is significant
        if best_likelihood > -np.log10(0.050000001):
            nta_gc = calc_epoch_granger_causality_ij(cut_data, i, j,
                                                     order_best_likelihood,
                                                     sfreq)  # from i-row to j-col
        else:
            print('No significance found...')
    
    return nta_gc, best_likelihood, order_best_likelihood


def export_to_csv(number_of_channels, data_durations, csv_column_names, channel_names, gc, max_likelihoods,
                  max_likelihoods_order):
    csv_data = csv_column_names
    
    for i in range(number_of_channels):
        for j in range(number_of_channels):
            for duration in data_durations:
                if max_likelihoods[i, j, duration] > -np.log10(0.05):
                    gc_val = gc[i, j, duration]
                    order = max_likelihoods_order[i, j, duration]
                    gc_times_likelihood = max_likelihoods[i, j, duration] * gc[i, j, duration]

                    csv_data.append([f'{channel_names[i]} - {channel_names[j]}'
                                        , gc_val, max_likelihoods[i, j, duration],
                                     order, duration, gc_times_likelihood])

    # File path
    file_path = r'C:/Users/samue/OneDrive/FindNeuro/Practice/Granger/ntaGrangerRealdata/nta_data.csv'

    # Write data to CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)


def export_to_excel(number_of_channels, data_durations, csv_column_names, channel_names, gc, max_likelihoods, max_likelihoods_order):
    csv_data = []

    for i in range(number_of_channels):
        for j in range(number_of_channels):
            for duration in data_durations:
                if max_likelihoods[i, j, duration] > -np.log10(0.05):
                    gc_val = gc[i, j, duration]
                    order = max_likelihoods_order[i, j, duration]
                    gc_times_likelihood = max_likelihoods[i, j, duration] * gc[i, j, duration]

                    csv_data.append([
                        f'{channel_names[i]} - {channel_names[j]}', 
                        gc_val, 
                        max_likelihoods[i, j, duration], 
                        order, 
                        duration, 
                        gc_times_likelihood
                    ])

    df = pd.DataFrame(csv_data, columns=csv_column_names)
    file_path = r'C:\Users\samue\OneDrive\FindNeuro\Practice\Granger\ntaGrangerRealdata\nta_data.xlsx'

    df.to_excel(file_path, index=False)
    
def export_sim_to_excel(number_of_channels, data_durations, csv_column_names, channel_names, gc, max_likelihoods, max_likelihoods_order):
    csv_data = []

    for i in range(number_of_channels):
        for j in range(number_of_channels):
            for duration in data_durations:
                if max_likelihoods[i, j, duration] > -np.log10(0.05):
                    gc_val = gc[i, j, duration]
                    order = max_likelihoods_order[i, j, duration]
                    gc_times_likelihood = max_likelihoods[i, j, duration] * gc[i, j, duration]

                    csv_data.append([
                        f'Channel {[i+1]} - Channel{[j+1]}', 
                        gc_val, 
                        max_likelihoods[i, j, duration], 
                        order, 
                        duration, 
                        gc_times_likelihood
                    ])

    df = pd.DataFrame(csv_data, columns=csv_column_names)
    file_path = r'C:\Users\samue\OneDrive\FindNeuro\Practice\Granger\ntaGrangerRealdata\nta_sim_data.xlsx'

    df.to_excel(file_path, index=False)

def event_critical_nodes(max_likelihoods, max_likelihood_gc_values, threshold=None):
    '''
    Calculates the criticality of the nodes per event
    :param max_likelihoods: the likelihood of the connection
    :param max_likelihood_gc_values: the GC value of the connection
    '''
    if threshold is None:
        threshold = -np.log10(0.05)
    sig_connections_indices = np.where(max_likelihoods.squeeze() > threshold)
    node_crit = np.zeros((max_likelihood_gc_values.shape[0]))
    conn_idx = []
    subset_of_nodes = set()
    for i, j in zip(sig_connections_indices[0], sig_connections_indices[1]):
        # GC value can be positive or negative depending on relationship - we (hopefully!) don't care about the sign
        node_crit[i] += abs(max_likelihood_gc_values[i, j])
        conn_idx.append((i, j))

        # connections.append({'val': con_data[i, j], 'label_from': label_names[i], 'label_to': label_names[j],
        #                    'pos': labels_info[j]['max_coord'].tolist()})

    return node_crit, conn_idx


def critical_node_comparison(max_likelihoods, max_likelihood_gc_values):
    ode_crit, edges = event_critical_nodes(max_likelihoods, max_likelihood_gc_values, threshold=None)
    G = nx.from_numpy_array(np.squeeze(max_likelihood_gc_values), create_using=nx.DiGraph)

    weighted_out_degree_centrality = {node: sum(weight for _, _, weight in G.out_edges(node, data='weight')) for
                                      node in G.nodes()}

    return ode_crit, weighted_out_degree_centrality


#%% creating/loading data
data, names, sfreq, ijs = load_electrode_data(data_path=r'C:\Users\samue\OneDrive\FindNeuro\Practice\Granger\ntaGrangerRealdata\electrodes_data_sub10_Seizure3_bipolar.npy',
                                          metadata_path=r'C:\Users\samue\OneDrive\FindNeuro\Practice\Granger\ntaGrangerRealdata\electrodes_meta_data_sub10_Seizure3_bipolar.npz')

data = data[[0, 7, 14, 15, 18, 21, 22, 29, 30, 34, 35, 38, 41, 42, 45, 48, 49, 54, 59, 60, 63, 66, 67, 70, 73, 74, 78, 84, 85, 92, 99, 100, 103, 106, 107, 111, 115, 116, 119, 122],:]
names = names[[0, 7, 14, 15, 18, 21, 22, 29, 30, 34, 35, 38, 41, 42, 45, 48, 49, 54, 59, 60, 63, 66, 67, 70, 73, 74, 78, 84, 85, 92, 99, 100, 103, 106, 107, 111, 115, 116, 119, 122]]

#resected channels are now 7 and 8


sim_data, sim_names = create_data(data, sfreq)
sim_data = force_causality(sim_data,noise = 1)

stat, failed_stat = stationarity_test(data)

#%% i,j specific gc test, finds max likelihd

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plotting real data
x_real = np.linspace(1, 4000, 4000)
axs[1].plot(x_real, data[7, :4000], label=names[7])
axs[1].plot(x_real, data[6, :4000], label=names[6])
axs[1].set_title('Real Data')
axs[1].legend()

# Plotting simulated data
x_sim = np.linspace(1, 40, 40)
axs[0].plot(x_sim, sim_data[1, :40], label='Channel 2: From')
axs[0].plot(x_sim, sim_data[5, :40], label='Channel 6: To')
axs[0].set_title('Simulated Data')
axs[0].legend()

plt.show()

#%% for real data

gc, likelihood, lk_order = nitime_granger(data, 
                                          sfreq=sfreq, 
                                          channel_names=names, 
                                          data_durations=[2], 
                                          maxlag=20)

#%%

gc_abs = abs(gc)

ode_crit, weighted_out_degree_centrality = critical_node_comparison(likelihood[:, :, 2], 
                                                                    gc_abs[:, :, 2])


#%% for simulated data

gc_sim, likelihood_sim, lk_order_sim = nitime_granger(sim_data, 
                                                      sfreq=sfreq, 
                                                      channel_names=names, 
                                                      data_durations=[2], 
                                                      maxlag=20)

ode_crit_sim, weighted_out_degree_centrality_sim = critical_node_comparison(likelihood_sim[:, :, 2], 
                                                                            gc_sim[:, :, 2])


#%%
export_sim_to_excel(number_of_channels=sim_data.shape[0], 
                data_durations=[2], 
                csv_column_names=['Channels', 'gc value', 'likelihood', 'order', 'data duration', 'gc * likelihood'], 
                channel_names=sim_names, 
                gc=gc_sim, 
                max_likelihoods=likelihood_sim, 
                max_likelihoods_order=lk_order_sim)



export_to_excel(number_of_channels=data.shape[0],
                data_durations=[2],
                csv_column_names=['Channels', 'gc value', 'likelihood', 'order', 'data duration', 'gc * likelihood'],
                channel_names=names,
                gc=gc,
                max_likelihoods=likelihood,
                max_likelihoods_order=lk_order)

print('done')
#%%

csv_file = r'C:/Users/samue/OneDrive/FindNeuro/Practice/Granger/ntaGrangerRealdata/likelihood.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(likelihood[:,:,2])
    
csv_file = r'C:/Users/samue/OneDrive/FindNeuro/Practice/Granger/ntaGrangerRealdata/order.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(lk_order[:,:,2])
        
#%% one ntagranger loop


#%%
"""

#%% one lrtest loop
i = 9
j = 30
lag = 1

series_to = data[j,:]
series_from = data[i,:]

series_to = np.diff(series_to)[1:]
series_from = np.diff(series_from)[1:]

verbose = False

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# Plotting real data
x_real = np.linspace(1, 7167, 7167)
axs[1].plot(x_real, series_to, label='series to')
axs[1].plot(x_real, series_from, label='series from')
axs[1].set_title('Real Data')
axs[1].legend()
plt.show()


print('1')
lrt = -np.log10(lrtest(data[i, :], data[j, :], lag)[1] + 0.00000001)
print('2')

x = pd.DataFrame(columns=['c1', 'c2'], data=zip(series_to, series_from))

# create lagmat of both time series
dta = lagmat2ds(x, lag, trim="both", dropex=1)  # must iterate through all lag

dtaown = add_constant(dta[:, 1: (lag + 1)], prepend=False)
dtajoint = add_constant(dta[:, 1:], prepend=False)
if (
        dtajoint.shape[1] == (dta.shape[1] - 1)
        or (dtajoint.max(0) == dtajoint.min(0)).sum() != 1
):
    raise InfeasibleTestError(
        "The x values include a column with constant values and so"
        " the test statistic cannot be computed."
    )
print('3')
#  Run ols on both models without and with lags of second variable
res2down = OLS(dta[:, 0], dtaown).fit()
res2djoint = OLS(dta[:, 0], dtajoint).fit()

# likelihood ratio test pvalue:
lr = -2 * (res2down.llf - res2djoint.llf)
if verbose:
    print(
        "likelihood ratio test: chi2=%-8.4f, p=%-8.4f, df=%d"
        % (lr, stats.chi2.sf(lr, lag), lag)
    )
result = (lr, stats.chi2.sf(lr, lag), lag)

gctests = grangercausalitytests(x, maxlag=20, verbose=False)
#%% exportingto csv
export_to_csv(data.shape[0], data_durations=[4],
              csv_column_names=['Channels', 'gc value', 'likelihood', 'order', 'data duration', 'gc * likelihood'],
              channel_names=names, gc=gc, max_likelihoods=likelihood, max_likelihoods_order=lk_order)


#%% i,j specific gc test, finds max likelihd
i = 29
j = 12

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
num_samples_plotted = 7000

rows_plotted = [i, j]
for k in rows_plotted:
    x = np.linspace(1, num_samples_plotted, num_samples_plotted)
    x = np.array(x)
    axs[0].plot(x, sim_data[k, :num_samples_plotted]) 
    axs[1].plot(x, data[k, :num_samples_plotted]) 

plt.show()

gc1a, gc1b, gc1c = nitime_granger_ij(sim_data, sfreq, i, j, maxlag=20)
gc2a, gc2b, gc2c = nitime_granger_ij(data, sfreq, i, j, maxlag=20)

gct = np.array([['sig>1.301','gc val', 'p val', 'order'],
                ['sim', gc1a, gc1b, gc1c],
                ['real', gc2a, gc2b, gc2c]])

#%% one ntagranger loop
i = 29
j = 17
order = 5
time_series = data
#pdata = tsu.percent_change(time_series)

ij_chans = time_series[[i, j]]

print('1')
time_series = ts.TimeSeries(ij_chans, sampling_interval=1 / sfreq)
print('2')
G = nta.GrangerAnalyzer(time_series, order=int(order))  # , ij=[(i, j)]
print('3')
freq_idx_G = np.arange(len(G.frequencies))
print('4')
gc_val = np.nanmean(G.causality_xy[0, 1, freq_idx_G], -1)
print('5')
gc_val = gc_val if not np.isnan(gc_val) else 0
print('done')
"""
