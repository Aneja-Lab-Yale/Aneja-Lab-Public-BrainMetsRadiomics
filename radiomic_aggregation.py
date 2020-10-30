# Radiomic Feature Aggregation
# By Aneja Lab - Enoch Chang

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------- #
# folders
# -------------------- #

ROOT = '/Users/enochchang/brainmets/'

# for saving aggregated radiomic features
UWA_radiomic_fp = ROOT + 'UWA_radiomic.csv'
WA_radiomic_fp = ROOT + 'WA_radiomic.csv'
big3_radiomic_fp = ROOT + 'big3_radiomic.csv'
big1_radiomic_fp = ROOT + 'big1_radiomic.csv'
smallest_radiomic_fp = ROOT + 'smallest_radiomic.csv'

# for creating survival object in R
surv_labels_fp = ROOT + 'surv_labels.csv'

# file for linked image IDs + clincial data
COHORT_fp = ROOT + 'linked_cohort.csv'

# --------------------- #
# load radiomic data
# --------------------- #

# this is the file containing radiomic features of all patients
allradiomics = pd.read_csv('/Users/enochchang/brainmets/radiomicsresults.csv')

# create list of columns starting with 'original' and 'wavelet'
value_feature_names = [x for x in allradiomics.columns if (x.startswith('original'))] + [x for x in allradiomics.columns if (x.startswith('wavelet'))]

# combine row_ID (patient identifier) + metnum for unique met-specific ID
allradiomics['row_ID'] = allradiomics['row_ID'].astype(str)
allradiomics['metnum'] = allradiomics['metnum'].astype(str)
allradiomics['ID'] = allradiomics[['row_ID', 'metnum']].apply(lambda x: '_'.join(x), axis=1)

# convert row_ID back to int to match row_ID from survival dataset
allradiomics['row_ID'] = allradiomics['row_ID'].astype(int)

allradiomics = allradiomics[['ID', 'metnum', 'row_ID'] + value_feature_names]

# clean column names
allradiomics.columns = ['ID', 'metnum', 'row_ID'] + [' '.join(x.split('original_')[-1].split('_')) for x in value_feature_names]

# keep names of radiomic features only
clean_col_names = allradiomics.columns.tolist()[3:]

# --------------------- #
# load survival data
# --------------------- #

# load cohort linked image ID + clinical data
surv = pd.read_csv(COHORT_fp, index_col=0)

# combine row_ID (patient identifier) + metnum for unique met-specific ID
surv['Row_ID'] = surv['Row_ID'].astype(str)
surv['metnum'] = surv['metnum'].astype(str)
surv['ID'] = surv[['Row_ID', 'metnum']].apply(lambda x: '_'.join(x), axis=1)

# create list of IDs to select cohort from allradiomics
surv5mmids = list(surv['ID'])

# restore row_ID to type int
surv['Row_ID'] = surv['Row_ID'].astype(int)

# create df of radiomic data containing only those in the desired subgroup
surv5mm_radiomic = allradiomics.loc[allradiomics['ID'].isin(surv5mmids)].copy()

# save labels mapped to ID (ensures they correspond since radiomic data is not ordered after parallel processing)
# ID for unique met specific data

timedic = pd.Series(surv['survival'].values, index=surv['Row_ID']).to_dict()
eventdic = pd.Series(surv['dead'].values, index=surv['Row_ID']).to_dict()
XYdic = pd.Series(surv['XY'].values, index=surv['ID']).to_dict()
Zdic = pd.Series(surv['Z'].values, index=surv['ID']).to_dict()

# map labels to radiomic data
surv5mm_radiomic['survival'] = surv5mm_radiomic['ID'].map(timedic)
surv5mm_radiomic['dead'] = surv5mm_radiomic['ID'].map(eventdic)
surv5mm_radiomic['XY'] = surv5mm_radiomic['ID'].map(XYdic)
surv5mm_radiomic['Z'] = surv5mm_radiomic['ID'].map(Zdic)

# -------------------- #
# calculate volume
# -------------------- #

# surv5mm_radiomic['volume'] = surv5mm_radiomic['XY'] * surv5mm_radiomic['XY'] * surv5mm_radiomic['Z']

# TODO FIX THIS
pi = 3.1415926535897931
r = surv5mm_radiomic['XY'] / 2  # XY is the diameter at the largest width of the tumor in the axial plane
surv5mm_radiomic['volume'] = 4/3 * pi * r**3

voldic = pd.Series(surv5mm_radiomic['volume'].values, index=surv5mm_radiomic['row_ID']).to_dict()

# -------------------- #
# Un-weighted Average (UWA)
# -------------------- #

# store shape features to be summed in a temporary df
sumdf = surv5mm_radiomic.iloc[:, 2:17].copy()
sumdf = sumdf.groupby('row_ID').sum()

# store columns to be averaged in a temporary df
# last 4 cols are surv, dead, XY, Z
meandf = surv5mm_radiomic.iloc[:, np.r_[2, 17:854]].copy()
meandf = meandf.groupby('row_ID').mean()

# combine sum and mean df
UWA = sumdf.join(meandf)

# save column names before renaming to numeric (for mRMRe)
col_names = {x:y for x,y in zip(UWA.columns, range(0,len(UWA.columns)))}

# rename columns to numeric (the format required for mRMRe)
UWA = UWA.rename(columns=col_names)

# standardize
UWA.columns = UWA.columns.astype(int)
scaler = StandardScaler()
UWA.iloc[:, :] = scaler.fit_transform(UWA.iloc[:, :])

# save radiomic features for feature selection
UWA.to_csv(UWA_radiomic_fp, index=True)

# -------------------- #
# create df for survival object in R
# -------------------- #

# can be used across all aggregation models since order is the same
UWA['Time'] = UWA.index.to_series().map(timedic)
UWA['Event'] = UWA.index.to_series().map(eventdic)
surv = UWA[['Time', 'Event']]
surv.to_csv(surv_labels_fp, index=False)

# -------------------- #
# save dictionaries
# -------------------- #
# restore column names
inv_col_names = {v: k for k, v in col_names.items()}

# save dictionaries
np.save(ROOT + 'colnames.npy', inv_col_names)
np.save(ROOT + 'timedic.npy', timedic)
np.save(ROOT + 'eventdic.npy', eventdic)
np.save(ROOT + 'voldic.npy', voldic)


# -------------------- #
# Weighted Average (WA)
# -------------------- #

"""
Weighted Average Calculation
1. calculate weight: (per tumor volume) / (total volume per patient)
2. multiply each per-tumor-value by weight
3. sum weighted per-tumor-values to obtain weighted average per patient
"""

# calculate total volume per patient
totalvol = surv5mm_radiomic[['row_ID', 'volume']].copy()
totalvol = totalvol.groupby('row_ID').sum()

# save labels mapped to ID
totalvoldic = pd.Series(totalvol['volume'].values, index=totalvol.index).to_dict()
np.save(ROOT + 'totalvoldic.npy', totalvoldic)

# map labels to radiomic data
surv5mm_radiomic['totalvolume'] = surv5mm_radiomic['row_ID'].map(totalvoldic)

# calculate weight as the per-tumor proportion of per-patient volume
surv5mm_radiomic['wt'] = surv5mm_radiomic['volume'] / surv5mm_radiomic['totalvolume']

# store shape features to be summed in a temporary df
sumdf = surv5mm_radiomic.iloc[:, 2:17].copy()
sumdf = sumdf.groupby('row_ID').sum()

# select all non shape columns + row_ID + wt
meandf = surv5mm_radiomic.iloc[:, np.r_[2, 17:854, 860]].copy()
meandf = meandf.set_index('row_ID')

# multiply all values by weight
meandf = meandf.iloc[:, 0:837].multiply(meandf['wt'], axis='index')

# sum the weighted values to get weighted average
meandf = meandf.groupby(meandf.index).sum()

# combine sum and mean df
WA = sumdf.join(meandf)

# rename columns to numeric (the format required for mRMRe)
WA = WA.rename(columns=col_names)

# standardize
WA.columns = WA.columns.astype(int)
scaler = StandardScaler()
WA.iloc[:, :] = scaler.fit_transform(WA.iloc[:, :])

# save radiomic features only for feature selection
WA.to_csv(WA_radiomic_fp, index=True)


# -------------------- #
# WA of 3 Largest
# -------------------- #

big3 = surv5mm_radiomic.copy()

# select 3 largest sorted by volume
big3 = big3.sort_values(by=['row_ID', 'volume'], ascending=False).groupby('row_ID').head(3)

# store shape features to be summed in a temporary df
sumdf = big3.iloc[:, 2:17].copy()
sumdf = sumdf.groupby('row_ID').sum()

# calculate weighted average of 3 largest
# calculate total volume of top 3 mets
big3totalvol = big3[['row_ID', 'volume']].copy()
big3totalvol = big3totalvol.groupby('row_ID').sum()

# save labels mapped to ID
big3totalvoldic = pd.Series(big3totalvol['volume'].values, index=big3totalvol.index).to_dict()

# maps labels to radiomic data
big3['big3totalvolume'] = big3['row_ID'].map(big3totalvoldic)

# calculate weight
big3['big3wt'] = big3['volume'] / big3['big3totalvolume']

# select all non shape columns + row_ID + wt
meandf = big3.iloc[:, np.r_[2, 17:854, 862]].copy()
meandf = meandf.set_index('row_ID')

# multiply all values by weight
meandf = meandf.iloc[:, 0:837].multiply(meandf['big3wt'], axis='index')

# sum the weighted values to get weighted average
meandf = meandf.groupby(meandf.index).sum()

# combine sum and mean df
big3 = sumdf.join(meandf)

# rename columns to numeric (the format required for mRMRe)
big3 = big3.rename(columns=col_names)

# standardize
big3.columns = big3.columns.astype(int)
scaler = StandardScaler()
big3.iloc[:, :] = scaler.fit_transform(big3.iloc[:, :])

# save radiomic features only for feature selection
big3.to_csv(big3_radiomic_fp, index=True)


# -------------------- #
# Largest
# -------------------- #

# note: no aggregation needed since selecting single met

big1 = surv5mm_radiomic.copy()

# select largest sorted by volume
big1 = big1.sort_values(by=['row_ID', 'volume'], ascending=False).groupby('row_ID').head(1)

print(big1['volume'].describe())
print(np.percentile(big1['volume'], 33))
print(np.percentile(big1['volume'], 66))

big1voldic = pd.Series(big1['volume'].values,index=big1['row_ID']).to_dict()
np.save(ROOT + 'big1voldic.npy', big1voldic)

# set row_ID as index
big1 = big1.set_index('row_ID')

# keep radiomic features only
big1 = big1.iloc[:, 2:853]

# rename columns to numeric (the format required for mRMRe)
big1 = big1.rename(columns=col_names)
big1.head(1)

# standardize
big1.columns = big1.columns.astype(int)
scaler = StandardScaler()
big1.iloc[:, :] = scaler.fit_transform(big1.iloc[:, :])

# save radiomic features only for feature selection
big1.to_csv(big1_radiomic_fp, index=True)

# -------------------- #
# Smallest
# -------------------- #

smallest = surv5mm_radiomic.copy()

# select smallest sorted by volume
smallest = smallest.sort_values(by=['row_ID', 'volume'], ascending=True).groupby('row_ID').head(1)

# set row_ID as index
smallest = smallest.set_index('row_ID')

# keep radiomic features only
smallest = smallest.iloc[:, 2:853]

# rename columns to numeric (the format required for mRMRe)
smallest = smallest.rename(columns=col_names)

# standardize
smallest.columns = smallest.columns.astype(int)
scaler = StandardScaler()
smallest.iloc[:, :] = scaler.fit_transform(smallest.iloc[:, :])

# save radiomic features only for feature selection
smallest.to_csv(smallest_radiomic_fp, index=True)