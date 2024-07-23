# Introduction
This work aims to demonstrate a process for quantifying a Granger Causality relationship between nodes in an ictal network, to contribute to a larger system used to find the central nodes that cause seizures.

Given two time series (in this case, two channels of sEEG data), a Granger Causality test will quantify a causal relationship between them. **I.E. if a change in one, is causing a change in the other.**


# Process

### Simulating Data
I want to start by creating some random data. The complete dataset contains 7 seconds at 1024hz. The code iterates through this line to create a time series.
```python
r.append(0.5 * r[-1] + np.random.rand())  # Append a new point: 50% of the last point plus a random value, simulating realistic, noisy data progression.
```
To test if the program can find causality, I create artificial, and predictable, causality in the data. 

Method 1: `copy`
First, a given channel is copied to another row, and some random noise is added. 
```python
independent_channel = 1
dependent_channel = 5
#  copy each datapoint and add noise
for sample in range(np.shape(data)[1]):
        forced_data[dependent_channel, sample] = data[independent_channel, sample] + (noise * (np.random.rand() - 0.5))
```
Next, the data is shifted by a certain amount of lags (datapoints)
```python
lag = 2
#  roll dataset by given lags
forced_data[dependent_channel, :] = np.roll(data[dependent_channel, :], lag)
```
The result is a straightforward causal relationship, where a change in the first channel(1) causes a change in the second channel (5), after a set number of lags(2). 

![Simulated vs Real Data](https://github.com/sbockfind/main/blob/main/Figure%202024-07-10%20192428.png)

Method 2: `Average`
Another method of forcing causality. Some of the information of the dependent channel was kept by averaging the dependent and independent channels, again adding noise and 'rolling' the data.

```python
independent_channel = 1
dependent_channel = 6
lag = 2
#  average the data points and add noise
for sample in range(np.shape(data)[1]):
  forced_data[dependent_channel, sample] = ((0.5 * data[dependent_channel, sample]) + 
                                            (0.5 * data[independent_channel, sample]) +
                                            (noise * (np.random.rand() - 0.5)))
#  roll data by given lags
forced_data[dependent_channel, :] = np.roll(data[dependent_channel, :], lag)
```
Using different methods for a few different pairs creates some variation in how obvious the forced causality is. In the chart above, the causality is very apparent (red circles), but other methods make it harder to point out. The goal is to test if the program can pick out causality among more noise. 


### Likelihood Ratio Test: `lrtest()`

This function is crucial in determining if one time series provides significant predictive power for another, beyond what is offered by its own past values. It is extracted from the statsmodels grangercausalitytests() function.


Two Models are defined - one without the lagged values of the second variable (dtaown) and one with them (dtajoint).
```python
dtaown = add_constant(dta[:, 1: (lag + 1)], prepend=False)
dtajoint = add_constant(dta[:, 1:], prepend=False)
```
OLS Models: Ordinary Least Squares (OLS) regression is run on both models, then the likelihood ratio test is computed, comparing the log-likelihoods of the two models.
```python
res2down = OLS(dta[:, 0], dtaown).fit()
res2djoint = OLS(dta[:, 0], dtajoint).fit()

lr = -2 * (res2down.llf - res2djoint.llf)
```
I am interested in the P-value, which indicates the statistical significance of the likelihood ratio. A low p-value (less than 0.05) suggests that the lagged values of the second time series provide significant predictive power for the first time series, indicating a Granger causality relationship. Granger Causality Calculation: The function calculates the Granger Causality value for the specified pair of channels.

### Granger Analyzer `calc_epoch_granger_causality_ij()`

The program uses the `GrangerAnalyzer` class from the `nitime` package to estimate the causality between different time series. This identifies how changes in one time series can predict changes in another.

The function creates a TimeSeries object from two channels, then runs GrangerAnalyzer for a specified lag order.

```python

ij_chans = time_series[[i, j]]
time_series = ts.TimeSeries(ij_chans, sampling_interval=1 / sfreq)

G = nta.GrangerAnalyzer(time_series, order=int(order))

gc_val = np.nanmean(G.causality_xy[0, 1, freq_idx_G], -1)
gc_val = gc_val if not np.isnan(gc_val) else 0
```
### Main Loop `nitime_granger()`

The function first cuts the data around the "seizure onset" to create the time window for analysis.

For each combination of channels, the function performs the likelihood ratio test. It also saves the lag at which the highest likelihood occurs.

```python
for lag in range(maxlag):
    lr = lrtest(cut_data[i, :], cut_data[j, :], lag)[1]
    lr = -np.log10(lr + 0.00000001) # add a small amount to account for zeros
    
    if lr > best_likelihood[i, j, duration]:
        best_likelihood[i, j, duration] = lr
        order_best_likelihood[i, j, duration] = lag
```

The function calculates the Granger Causality value if the p-value from the likelihood ratio test `lr` is significant.

```python
if best_likelihood[i, j, duration] > -np.log10(0.050000001):
    nta_gc[i, j, duration] = calc_epoch_granger_causality_ij(cut_data, i, j,
                                                             order_best_likelihood[i, j, duration],
                                                             sfreq)
```
The program outputs data for the Granger values, maximum likelihoods, and the lag at which they occur.

# Output Analysis
The output of the Granger Causality analysis is crucial in validating the effectiveness of the code. The results demonstrate that the program can accurately identify the forced causality relationships set by the user. By analyzing the output matrices, we can observe how well the code picks up on the artificial causal links introduced in the data, even amidst varying levels of noise. This section will delve into the details of interpreting the output and assessing the accuracy of the causality detection.


For testing simulated data, causality is forced using method 1 (copy) from channels 1 to 5, and method 2 (average) from channels 1 to 6. Method 2 is used twice from channels 3 to 7. Three total combinations are forced

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/sbockfind/main/blob/main/sim_data_likelihood.png" alt="Likelihood" width="45%">
    <img src="https://github.com/sbockfind/main/blob/main/sim_data_order.png" alt="Order" width="45%">
</div>
The results show that the program can accurately identify the forced causality relationships. 

![Granger Causality Value](https://github.com/sbockfind/main/blob/main/sim_data_gc.png)


