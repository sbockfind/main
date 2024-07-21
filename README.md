# Heading 1
## Heading 2
### Heading 3

**Bold Text**

*Italic Text*

- List item 1
- List item 2
- List item 3

[Link to GitHub](https://github.com/)

![Image Alt Text](https://via.placeholder.com/150)

# Code block
print("Hello, World!")

# Introduction
This work aims to demonstrate a process for quantifying a Granger Causality relationship between nodes in an ictal network, to contribute to a larger system used to find the central nodes that cause seizures.

Given two time series (in this case, two channels of sEEG data), a Granger Causality test will quantify a causal relationship between them. **I.E. if a change in one, is causing a change in the other.**


# Process

## Simulating Data
I want to start by creating some random data. The complete dataset contains 7 seconds at 1024hz. The code iterates through this line to create a time series.
```python
r.append(0.5 * r[-1] + np.random.rand())  # Append a new point: 50% of the last point plus a random value, simulating realistic, noisy data progression.
```
To test if the program can find causality, I create artificial, and predictable, causality in the data. 

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


## Likelihood Ratio Test: lrtest()

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
I am interested in the P-value, which indicates the statistical significance of the likelihood ratio. A low p-value (less than 0.05) suggests that the lagged values of the second time series provide significant predictive power for the first time series, indicating a Granger causality relationship.

## Granger Analyzer

The main loop of the program runs data through a function to determine 



The data is first cropped down to a few channels from each electrode (e.g. LFIa01-LFIa02, LFIa08-LFIa09, and LFIa15-LFIa16) and 4 seconds of data: 2 seconds on either side of the seizure onset. The cropped data contains the resected area corresponding to: RFM08-RFM09, RFM09-RFM10. Each combination of channels is run through a Likelihood Ratio (LR) test, pulled out of the statsmodels grangercausalitytests() function. Only the highest likelihood and the order (or lag) at which it occurs are kept.
	When the highest likelihood is significant at 95% confidence, a GC test is run for that combination at the order recorded. (the function used is calc_epoch_granger_causality_ij(), which uses the nitime GrangerAnalyzer())
	The program then compares centrality measures event_critical_nodes() and nx’s weighted_out_degree_centrality.
