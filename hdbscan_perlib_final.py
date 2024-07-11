#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
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

cluster_sizes = {x : [] for x in bincounts}

for i in range(0, len(bincounts)):

    # Read metadata
    df_meta = pd.read_csv(metadata_file, index_col=0)
    
    # Split spectra into groups by library
    tetrad = df_meta[df_meta['library'].isin(['tetrad', 'tetrad, 17.3 loop', 'tetrad, 17.4 loop', 'tetrad, 17.10 loop'])].index.tolist()
    loop_17_3 = df_meta[df_meta['library'].isin(['17.3 loop', 'tetrad, 17.3 loop'])].index.tolist()
    loop_17_4 = df_meta[df_meta['library'].isin(['17.4 loop', 'tetrad, 17.4 loop'])].index.tolist()
    loop_17_10 = df_meta[df_meta['library'].isin(['17.10 loop', 'tetrad, 17.10 loop'])].index.tolist()
    
    # Read binned_spectra
    df = pd.read_csv(binned_spectra_files[i], index_col=0)
    df.columns = [int(x) for x in df.columns]
    df = df[sorted(df.columns.tolist())]
    df_t = df.transpose()
    
    # 17.10
    X = df_t.loc[list(set(df_t.index.tolist()).intersection(set(loop_17_10)))]
    clustering_hdb = hdbscan.HDBSCAN()
    clustering_hdb.fit(X)
    df_cres = pd.DataFrame({'sample':X.index, 'c':clustering_hdb.labels_})
    df_cres.to_csv('cluster_assignments_17_10_' + str(bincounts[i]) + 'bins.csv')
    
    cs = pd.crosstab(df_cres['sample'], df_cres['c']).sum()   
    cluster_sizes[bincounts[i]].append(cs)


# Tetrad
X = df_t.loc[list(set(df_t.index.tolist()).intersection(set(tetrad)))]
clustering_hdb = hdbscan.HDBSCAN()
clustering_hdb.fit(X)
df_cres = pd.DataFrame({'sample':X.index, 'c':clustering_hdb.labels_})
df_cres.to_csv('cluster_assignments_tetrad_' + str(bincounts[i]) + 'bins.csv')

# 17.3
X = df_t.loc[list(set(df_t.index.tolist()).intersection(set(loop_17_3)))] 
clustering_hdb = hdbscan.HDBSCAN()
clustering_hdb.fit(X)
df_cres = pd.DataFrame({'sample':X.index, 'c':clustering_hdb.labels_})
df_cres.to_csv('cluster_assignments_17_3_' + str(bincounts[i]) + 'bins.csv')


# 17.4
X = df_t.loc[list(set(df_t.index.tolist()).intersection(set(loop_17_4)))]
clustering_hdb = hdbscan.HDBSCAN()
clustering_hdb.fit(X)
df_cres = pd.DataFrame({'sample':X.index, 'c':clustering_hdb.labels_})
df_cres.to_csv('cluster_assignments_17_4_' + str(bincounts[i]) + 'bins.csv')

# 17.10
X = df_t.loc[list(set(df_t.index.tolist()).intersection(set(loop_17_10)))]
clustering_hdb = hdbscan.HDBSCAN()
clustering_hdb.fit(X)
df_cres = pd.DataFrame({'sample':X.index, 'c':clustering_hdb.labels_})
df_cres.to_csv('cluster_assignments_17_10_' + str(bincounts[i]) + 'bins.csv')

