#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import hdbscan

# Filenames 

metadata_file = '../metadata/metadata.csv'
binned_spectra_files = '../spectra_prep/spectra_merged_trimmed_normalized.csv'

binned_spectra_files = [
    '../spectra_prep/spectra_merged_trimmed_normalized_binned_50.csv',
    '../spectra_prep/spectra_merged_trimmed_normalized_binned_100.csv',
    '../spectra_prep/spectra_merged_trimmed_normalized_binned_200.csv',
    '../spectra_prep/spectra_merged_trimmed_normalized_binned_400.csv',
    '../spectra_prep/spectra_merged_trimmed_normalized_binned_800.csv',
    '../spectra_prep/spectra_merged_trimmed_normalized_binned_1600.csv',
]

out_files = [
    'cluster_assignments_50_bins.csv',
    'cluster_assignments_100_bins.csv',
    'cluster_assignments_200_bins.csv',
    'cluster_assignments_400_bins.csv',
    'cluster_assignments_800_bins.csv',
    'cluster_assignments_1600_bins.csv',
]

bincounts = ['50', '100', '200', '400', '800', '1600']

# Read metadata
df_meta = pd.read_csv(metadata_file, index_col=0)

for i in range(0, len(bincounts)):

    # Read binned_spectra
    df = pd.read_csv(binned_spectra_files[i], index_col=0)
    df.columns = [int(x) for x in df.columns]
    df = df[sorted(df.columns.tolist())]
    df_t = df.transpose()
    
    # Here I remove all rows from the meta table which were filtered out in preprocessing
    df_meta = df_meta.loc[df.columns]
    
    # Read UNbinned spectra for plotting
    #df_raw = pd.read_csv(unbinned_spectra_file, index_col=0)
    #df_raw = df_raw.set_index('shift')
    #df_raw.columns = [int(x) for x in df_raw.columns]
    #df_raw = df_raw[sorted(df_raw.columns)]
    
    X = df_t
    clustering_hdb = hdbscan.HDBSCAN()
    clustering_hdb.fit(X)
    df_cres = pd.DataFrame({'sample':X.index, 'c':clustering_hdb.labels_})
    
    #for c in sorted(df_cres['c'].unique()):
    #    df_raw[df_cres[df_cres['c'] == c]['sample'].tolist()].plot(legend=False, figsize=(15, 5), title='cluster'+str(c))
        
    df_cres[['sample', 'c']].to_csv(out_files[i])

# Read metadata
df_meta = pd.read_csv(metadata_file, index_col=0)

# Read binned_spectra
df = pd.read_csv(binned_spectra_file, index_col=0)
df.columns = [int(x) for x in df.columns]
df = df[sorted(df.columns.tolist())]
df_t = df.transpose()

# Here I remove all rows from the meta table which were filtered out in preprocessing
df_meta = df_meta.loc[df.columns]

# Read UNbinned spectra for plotting
df_raw = pd.read_csv(unbinned_spectra_file, index_col=0)
df_raw = df_raw.set_index('shift')
df_raw.columns = [int(x) for x in df_raw.columns]
df_raw = df_raw[sorted(df_raw.columns)]

X = df_t
clustering_hdb = hdbscan.HDBSCAN()
clustering_hdb.fit(X)
df_cres = pd.DataFrame({'sample':X.index, 'c':clustering_hdb.labels_})

# The results of the rest of this script are not included in the manuscript

#for c in sorted(df_cres['c'].unique()):
#    df_raw[df_cres[df_cres['c'] == c]['sample'].tolist()].plot(legend=False, figsize=(15, 5), title='cluster'+str(c))
    
df_cres[['sample', 'c']].to_csv(out_file)


X_tsne = TSNE(n_components=2).fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X)

df_cres['pca_1'] = X_pca[:,0]
df_cres['pca_2'] = X_pca[:,1]
df_cres['tsne_1'] = X_tsne[:,0]
df_cres['tsne_2'] = X_tsne[:,1]
df_cres['lib'] = [df_meta.loc[x]['library'] for x in df_cres['sample']]

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(data=df_cres, x='pca_1', y='pca_2', hue='c', palette='tab10')
plt.subplot(2, 2, 2)
sns.scatterplot(data=df_cres, x='pca_1', y='pca_2', hue='lib', palette='tab10')
plt.subplot(2, 2, 3)
sns.scatterplot(data=df_cres, x='tsne_1', y='tsne_2', hue='c', palette='tab10')
plt.subplot(2, 2, 4)
sns.scatterplot(data=df_cres, x='tsne_1', y='tsne_2', hue='lib', palette='tab10')
plt.show()

