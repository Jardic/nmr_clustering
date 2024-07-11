#!/usr/bin/env python
# coding: utf-8

### Preprocessing of 1-D NMR spectra for clustering

import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt


# ---
# ### Data merge
# Fist, spectra are just merged from multiple files into one dataframe.

data_dir = '../raw_data/'

dataset_raw = {}
for f in os.scandir(data_dir):
    if f.is_file():
        f = f.name
        sp_name = int(f.split('.')[0])
        df_raw = pd.read_csv(data_dir + f, skiprows=1)
        df_raw = df_raw.iloc[:,[1, 3]]
        df_raw.columns=['intensity', 'shift']
        dataset_raw[sp_name] = df_raw['intensity'].tolist()

dataset_raw['shift'] = df_raw['shift']
df_raw = pd.DataFrame(dataset_raw)
samples = df_raw.columns.tolist()[:-1]

# Here is what all the spectra look in one plot. The beginnings and ends of them are useless for clustering but we will use a part of this region (between 9.7 and 10.2) to calculate the mean and standard deviation of the baseline. These will later be used to filter out dead spectra. 

df_raw.plot(x = 'shift', y=samples, legend=False, figsize=(15, 5))
plt.axvline(10.3, linewidth=1)
plt.axvline(12, linewidth=1)
plt.show()
#plt.savefig('all_spectra_before_prep.png', dpi=200)

# The baseline region visualised:

df_baseline = df_raw[df_raw['shift'].between(9.7, 10.2)].copy()
df_baseline.plot(x = 'shift', y=samples, legend=False, figsize=(15, 5))
plt.show()

# ---
# ### Spectra trimming

df_raw = df_raw[df_raw['shift'].between(10.3, 12)].copy()
df_raw.to_csv('spectra_merged_trimmed.csv')

# Trimmed spectra:

df_raw.plot(x='shift', y=samples, legend=False, figsize=(15, 5))
plt.show()

##### In the case that normalization is not applied, this piece of code should be used to adjust multiple-scanned spectra intensity values.
#df_meta = pd.read_csv('metadata.csv', index_col=0)
#multiscanned_names = df_meta[df_meta['multiscanned'] == True].index
#df_raw[multiscanned_names] = df_raw[multiscanned_names] / 4

# ---
# ### Detecting dead spectra
# Here I use the baseline region to calculate stsandard mean and standard deviation for each spectrum separately and then use those to decide, if a spectrum contains reasonable peaks or not.

q = 3.5
print('For this preprocessing run,', q, ' standard deviations were used.')

baseline_means = df_baseline.mean()
baseline_stds = df_baseline.std()

spectra_dead = []
spectra_good = []

for s in df_raw.columns[:-1]:
    if any(df_raw[s] >= baseline_means[s] + q*baseline_stds[s]):
        spectra_good.append(s)
    else:
        spectra_dead.append(s)

df_spectra_good = df_raw[spectra_good + ['shift']]
df_spectra_dead = df_raw[spectra_dead + ['shift']]


# **Good spectra:**
df_spectra_good.plot(x='shift', y=spectra_good, legend=False, figsize=(15, 5))
plt.show()

# **Bad spectra:**
df_spectra_dead.plot(x='shift', y=spectra_dead, legend=False, figsize=(15, 5))
plt.show()

print('Number of good spectra: ', len(df_spectra_good.columns)-1)
print('Number of dead spectra: ', len(df_spectra_dead.columns)-1)
df_raw = df_spectra_good

# ---
# ### Normalization
# min-max normalization.

from sklearn.preprocessing import MinMaxScaler
df_raw.columns = [str(x) for x in df_raw.columns]

scaler = MinMaxScaler()
scaler.fit(df_raw)
df_raw_normalized = scaler.transform(df_raw)
df_raw_normalized = pd.DataFrame(df_raw_normalized, index=df_raw.index, columns=df_raw.columns)

df_raw = df_raw_normalized

# **Good spectra normalized**

df_raw.plot(x='shift', y=samples, legend=False, figsize=(15, 5))
plt.show()
df_raw.to_csv('spectra_merged_trimmed_unfiltered_normalized.csv')


# ---
# ### Binning
# Next step is binning. Without binning we'd have 7000 dimensions to work with even after trimming the shift axis and so the point of binning is to reduce the number of dimensions enough to have a more reasonable number of dimensions, but not too much so that the useful information is lost. To achieve this we try a couple of binnings

df_raw = pd.read_csv('spectra_merged_trimmed_normalized.csv', index_col=0)
df_raw.shape

# Create a dataset with a given binning

bins_values = [25, 50, 100, 200, 400, 800, 1600]
binned = []

for bins in bins_values:

    df_raw['bin'] = pd.qcut(df_raw['shift'], q=bins)
    df_binned = df_raw.groupby(by='bin').mean().reset_index()
    del df_binned['bin']
    df_binned['shift'] = df_binned['shift'].round(decimals=4)
    df_binned = df_binned.set_index('shift')
    
    df_binned.to_csv('spectra_merged_trimmed_unfiltered_binned_' + str(bins) + '.csv')
