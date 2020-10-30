# Selected Radiomic Feature Loading
# By Aneja Lab - Enoch Chang

"""
Load and format radiomic features selected via mRMRe
"""

import numpy as np
import pandas as pd

# -------------------- #
# Folders
# -------------------- #

ROOT = '/Users/enochchang/brainmets/'

# all aggregated features (from radiomic_aggregation.py)
UWA_radiomic_fp = ROOT + 'UWA_radiomic.csv'
WA_radiomic_fp = ROOT + 'WA_radiomic.csv'
big3_radiomic_fp = ROOT + 'big3_radiomic.csv'
big1_radiomic_fp = ROOT + 'big1_radiomic.csv'
smallest_radiomic_fp = ROOT + 'smallest_radiomic.csv'

# mRMRe output
UWA_mrmre_fp = ROOT + 'surv_UWA_mRMRsolutions.csv'
WA_mrmre_fp = ROOT + 'surv_WA_mRMRsolutions.csv'
big3_mrmre_fp = ROOT + 'surv_big3_mRMRsolutions.csv'
big1_mrmre_fp = ROOT + 'surv_big1_mRMRsolutions.csv'
smallest_mrmre_fp = ROOT + 'surv_smallest_mRMRsolutions.csv'

# filenames for saving selected features
UWA_top40_fp = ROOT + 'UWA_top40.csv'
WA_top40_fp = ROOT + 'WA_top40.csv'
big3_top40_fp = ROOT + 'big3_top40.csv'
big1_top40_fp = ROOT + 'big1_top40.csv'
smallest_top40_fp = ROOT + 'smallest_top40.csv'

# -------------------- #
# load files, col names, and time/event data
# -------------------- #

# load files
UWA = pd.read_csv(UWA_radiomic_fp, index_col=0)
WA = pd.read_csv(WA_radiomic_fp, index_col=0)
big3 = pd.read_csv(big3_radiomic_fp, index_col=0)
big1 = pd.read_csv(big1_radiomic_fp, index_col=0)
smallest = pd.read_csv(smallest_radiomic_fp, index_col=0)

inv_col_names = np.load(ROOT + 'colnames.npy', allow_pickle=True).item()
timedic = np.load(ROOT + 'timedic.npy', allow_pickle=True).item()
eventdic = np.load(ROOT + 'eventdic.npy', allow_pickle=True).item()
nummetsdic = np.load(ROOT + 'nummetsdic.npy', allow_pickle=True).item()
voldic = np.load(ROOT + 'voldic.npy', allow_pickle=True).item()

# -------------------- #
# UWA
# -------------------- #

# read in top 40 features from mRMRe
UWA_top40 = pd.read_csv(UWA_mrmre_fp)
UWA_top40.head()
UWA_top40 = UWA_top40['X1'].tolist()

# select top 40 features from full set
UWA.columns = UWA.columns.astype(int)
UWA_mRMRe_top40 = UWA[UWA_top40].copy()

# map labels to radiomic data
UWA_mRMRe_top40['Time'] = UWA_mRMRe_top40.index.to_series().map(timedic)
UWA_mRMRe_top40['Event'] = UWA_mRMRe_top40.index.to_series().map(eventdic)
UWA_mRMRe_top40 = UWA_mRMRe_top40.rename(columns=inv_col_names)
UWA_mRMRe_top40.head()

# save
UWA_mRMRe_top40.to_csv(UWA_top40_fp, index=True)

# -------------------- #
# WA
# -------------------- #

# read in top 40 features from mRMRe
WA_top40 = pd.read_csv(WA_mrmre_fp)

WA_top40 = WA_top40['X1'].tolist()

# select top 40 features from full set
WA.columns = WA.columns.astype(int)
WA_mRMRe_top40 = WA[WA_top40].copy()


# map labels to radiomic data
WA_mRMRe_top40['Time'] = WA_mRMRe_top40.index.to_series().map(timedic)
WA_mRMRe_top40['Event'] = WA_mRMRe_top40.index.to_series().map(eventdic)
WA_mRMRe_top40 = WA_mRMRe_top40.rename(columns=inv_col_names)
WA_mRMRe_top40.head()

# save
WA_mRMRe_top40.to_csv(WA_top40_fp, index=True)

# -------------------- #
# WA of 3 Largest
# -------------------- #
# read in top 40 features from mRMRe
big3_top40 = pd.read_csv(big3_mrmre_fp)

big3_top40 = big3_top40['X1'].tolist()

# select top 40 features from full set
big3.columns = big3.columns.astype(int)
big3_mRMRe_top40 = big3[big3_top40].copy()

# map labels to radiomic data
big3_mRMRe_top40['Time'] = big3_mRMRe_top40.index.to_series().map(timedic)
big3_mRMRe_top40['Event'] = big3_mRMRe_top40.index.to_series().map(eventdic)
big3_mRMRe_top40 = big3_mRMRe_top40.rename(columns=inv_col_names)
big3_mRMRe_top40.head()

# save
big3_mRMRe_top40.to_csv(big3_top40_fp, index=True)

# -------------------- #
# Largest
# -------------------- #
# read in top 40 features from mRMRe
big1_top40 = pd.read_csv(big1_mrmre_fp)

big1_top40 = big1_top40['X1'].tolist()

# select top 40 features from full set
big1.columns = big1.columns.astype(int)
big1_mRMRe_top40 = big1[big1_top40].copy()

# map labels to radiomic data
big1_mRMRe_top40['Time'] = big1_mRMRe_top40.index.to_series().map(timedic)
big1_mRMRe_top40['Event'] = big1_mRMRe_top40.index.to_series().map(eventdic)
big1_mRMRe_top40 = big1_mRMRe_top40.rename(columns=inv_col_names)
big1_mRMRe_top40.head()

# map number of mets to radiomic data
big1_mRMRe_top40['nummets'] = big1_mRMRe_top40.index.to_series().map(nummetsdic)

# save
big1_mRMRe_top40.to_csv(big1_top40_fp, index=True)

# -------------------- #
# Smallest
# -------------------- #
# read in top 40 features from mRMRe
smallest_top40 = pd.read_csv(smallest_mrmre_fp)

smallest_top40 = smallest_top40['X1'].tolist()

# select top 40 features from full set
smallest.columns = smallest.columns.astype(int)
smallest_mRMRe_top40 = smallest[smallest_top40].copy()

# map labels to radiomic data
smallest_mRMRe_top40['Time'] = smallest_mRMRe_top40.index.to_series().map(timedic)
smallest_mRMRe_top40['Event'] = smallest_mRMRe_top40.index.to_series().map(eventdic)
smallest_mRMRe_top40 = smallest_mRMRe_top40.rename(columns=inv_col_names)
smallest_mRMRe_top40.head()

# save
smallest_mRMRe_top40.to_csv(smallest_top40_fp, index=True)