# Comparison of Survival Models
# By Aneja Lab - Enoch Chang

"""
Models:
- Cox Proportional Hazards
- LASSO-Cox
- Random Survival Forest

Aggregation Methods:
- Unweighted Average
- Weighted Average
- Weighted Average of 3 Largest
- Largest
- Largest with Number of Metastases
- Smallest

Sub-Analysis:
- Number of Metastases
- Volume of Largest Metastasis
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split


# -------------------- #
# folders
# -------------------- #

ROOT = '/Users/enochchang/brainmets/'

# selected features
UWA_top40_fp = ROOT + 'UWA_top40.csv'
WA_top40_fp = ROOT + 'WA_top40.csv'
big3_top40_fp = ROOT + 'big3_top40.csv'
big1_top40_fp = ROOT + 'big1_top40.csv'
smallest_top40_fp = ROOT + 'smallest_top40.csv'

# variables for sub-analysis
nummetsdic = np.load(ROOT + 'nummetsdic.npy', allow_pickle=True).item()
big1voldic = np.load(ROOT + 'big1voldic.npy', allow_pickle=True).item()


# -------------------- #
# Model Functions
# -------------------- #

def CPH_bootstrap(fp, num=False, sub=None):
	'''
	Compute CPH with bootstrapping

	:param fp: (str) filename of selected features
	:param num: (bool) set True to include number of mets (col 42)
	:param sub: (none) for sub-analysis, if it is specified, df is already created, do not need to load file
	:return: (str) C-index (95% confidence interval)
	'''

	if sub is not None:
		df = sub
	else:
		df = pd.read_csv(fp, index_col=0)

	# df = pd.read_csv(fp, index_col=0)

	# configure bootstrap (sampling 50% of data)
	n_iterations = 100
	n_size = int(len(df) * 0.50)

	# calculate population of statistics
	metrics = []
	for i in range(n_iterations):
		# prepare sample

		# if indicated, include number of mets (col 42)
		if num:
			sample = resample(df.iloc[:, np.r_[:20, 40, 41, 42]], n_samples=n_size)

		else:
			sample = resample(df.iloc[:, np.r_[:20, 40, 41]], n_samples=n_size)

		# calculate c-index and append to list
		cph = CoxPHFitter().fit(sample, 'Time', 'Event')
		score = concordance_index(sample['Time'], -cph.predict_partial_hazard(sample), sample['Event'])
		metrics.append(score)

	# calculate confidence interval
	alpha = 0.95
	p = ((1.0 - alpha) / 2.0) * 100
	lower = max(0.0, np.percentile(metrics, p))
	p = (alpha + ((1.0 - alpha) / 2.0)) * 100
	upper = min(1.0, np.percentile(metrics, p))
	med = np.percentile(metrics, 50)

	# identify aggregation method name
	if num:
		name = fp.split('/')[-1].split('_')[0] + ' + NumMets'
	else:
		name = fp.split('/')[-1].split('_')[0]

	return print(name, 'CPH', '%.3f (%.3f-%.3f)' % (med, lower, upper))


def LASSO_COX_bootstrap(fp, num=False):
	df = pd.read_csv(fp, index_col=0)

	# configure bootstrap (sampling 50% of data)
	n_iterations = 100
	n_size = int(len(df) * 0.50)

	# calculate population of statistics
	metrics = []
	for i in range(n_iterations):
		# prepare sample

		# if indicated, include number of mets (col 42)
		if num:
			sample = resample(df.iloc[:, np.r_[:20, 40, 41, 42]], n_samples=n_size)
			X = sample.iloc[:, np.r_[:20, 42]].copy()

		else:
			sample = resample(df.iloc[:, np.r_[:20, 40, 41]], n_samples=n_size)
			X = sample.iloc[:, :20].copy()

		X = X.to_numpy()
		y = sample[['Event', 'Time']].copy()
		y['Event'] = y['Event'].astype('bool')
		y = y.to_records(index=False)

		estimator = CoxnetSurvivalAnalysis(l1_ratio=1, alphas=[0.001])
		estimator.fit(X, y)
		score = estimator.score(X, y)
		metrics.append(score)

	# calculate confidence interval
	alpha = 0.95
	p = ((1.0 - alpha) / 2.0) * 100
	lower = max(0.0, np.percentile(metrics, p))
	p = (alpha + ((1.0 - alpha) / 2.0)) * 100
	upper = min(1.0, np.percentile(metrics, p))
	med = np.percentile(metrics, 50)

	# identify aggregation method name
	if num:
		name = fp.split('/')[-1].split('_')[0] + ' + NumMets'
	else:
		name = fp.split('/')[-1].split('_')[0]

	return print(name, 'Lasso-Cox', '%.3f (%.3f-%.3f)' % (med, lower, upper))


def RSF_bootstrap(fp, num=False):
	df = pd.read_csv(fp, index_col=0)

	# configure bootstrap (sampling 50% of data)
	n_iterations = 100
	n_size = int(len(df) * 0.50)

	# parameters
	NUMESTIMATORS = 100
	TESTSIZE = 0.20
	random_state = 20

	# calculate population of statistics
	metrics = []
	for i in range(n_iterations):
		# prepare sample

		# if indicated, include number of mets (col 42)
		if num:
			sample = resample(df.iloc[:, np.r_[:20, 40, 41, 42]], n_samples=n_size)
			X = sample.iloc[:, np.r_[:20, 42]].copy()

		else:
			sample = resample(df.iloc[:, np.r_[:20, 40, 41]], n_samples=n_size)
			X = sample.iloc[:, :20].copy()

		X = X.to_numpy().astype('float64')
		y = sample[['Event', 'Time']].copy()
		y['Event'] = y['Event'].astype('bool')
		y['Time'] = y['Time'].astype('float64')
		y = y.to_records(index=False)

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TESTSIZE, random_state=random_state)
		rsf = RandomSurvivalForest(n_estimators=NUMESTIMATORS,
								   min_samples_split=15,
								   min_samples_leaf=8,
								   max_features="sqrt",
								   n_jobs=-1,
								   random_state=random_state)
		rsf.fit(X_train, y_train)

		score = rsf.score(X_test, y_test)
		metrics.append(score)

		# calculate confidence interval
		alpha = 0.95
		p = ((1.0 - alpha) / 2.0) * 100
		lower = max(0.0, np.percentile(metrics, p))
		p = (alpha + ((1.0 - alpha) / 2.0)) * 100
		upper = min(1.0, np.percentile(metrics, p))
		med = np.percentile(metrics, 50)

		# identify aggregation method name
		if num:
			name = fp.split('/')[-1].split('_')[0] + ' + NumMets'
		else:
			name = fp.split('/')[-1].split('_')[0]

	return print(name, 'RSF', '%.3f (%.3f-%.3f)' % (med, lower, upper))

#%%
# ------------------------ #
# Cox Proportional Hazards
# ------------------------ #
CPH_bootstrap(UWA_top40_fp)
CPH_bootstrap(WA_top40_fp)
CPH_bootstrap(big3_top40_fp)
CPH_bootstrap(big1_top40_fp)
CPH_bootstrap(big1_top40_fp, num=True)
CPH_bootstrap(smallest_top40_fp)

#%%
# ---------------------- #
# Lasso-Cox
# ---------------------- #
LASSO_COX_bootstrap(UWA_top40_fp)
LASSO_COX_bootstrap(WA_top40_fp)
LASSO_COX_bootstrap(big3_top40_fp)
LASSO_COX_bootstrap(big1_top40_fp)
LASSO_COX_bootstrap(big1_top40_fp, num=True)
LASSO_COX_bootstrap(smallest_top40_fp)

# ----------------------- #
# Random Survival Forest
# ----------------------- #
RSF_bootstrap(UWA_top40_fp)
RSF_bootstrap(WA_top40_fp)
RSF_bootstrap(big3_top40_fp)
RSF_bootstrap(big1_top40_fp)
RSF_bootstrap(big1_top40_fp, num=True)
RSF_bootstrap(smallest_top40_fp)

# -------------------- #
# Sub-Analysis: Num Mets < 5, 5-10, 11+
# -------------------- #

df = pd.read_csv(UWA_top40_fp, index_col=0)

# map labels to radiomic data
df['nummets'] = df.index.to_series().map(nummetsdic)
df['nummets'] = df['nummets'].astype(int)

# create sub-groups
df_5 = df.loc[df['nummets'] < 5].copy()
df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
df_11 = df.loc[df['nummets'] >= 11].copy()

CPH_bootstrap(UWA_top40_fp, sub=df_5)
CPH_bootstrap(UWA_top40_fp, sub=df_5_10)
CPH_bootstrap(UWA_top40_fp, sub=df_11)


df = pd.read_csv(WA_top40_fp, index_col=0)

# map labels to radiomic data
df['nummets'] = df.index.to_series().map(nummetsdic)
df['nummets'] = df['nummets'].astype(int)

# create sub-groups
df_5 = df.loc[df['nummets'] < 5].copy()
df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
df_11 = df.loc[df['nummets'] >= 11].copy()

CPH_bootstrap(WA_top40_fp, sub=df_5)
CPH_bootstrap(WA_top40_fp, sub=df_5_10)
CPH_bootstrap(WA_top40_fp, sub=df_11)

df = pd.read_csv(big3_top40_fp, index_col=0)

# map labels to radiomic data
df['nummets'] = df.index.to_series().map(nummetsdic)
df['nummets'] = df['nummets'].astype(int)

# create sub-groups
df_5 = df.loc[df['nummets'] < 5].copy()
df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
df_11 = df.loc[df['nummets'] >= 11].copy()

CPH_bootstrap(big3_top40_fp, sub=df_5)
CPH_bootstrap(big3_top40_fp, sub=df_5_10)
CPH_bootstrap(big3_top40_fp, sub=df_11)

df = pd.read_csv(big1_top40_fp, index_col=0)

# map labels to radiomic data
df['nummets'] = df.index.to_series().map(nummetsdic)
df['nummets'] = df['nummets'].astype(int)

# create sub-groups
df_5 = df.loc[df['nummets'] < 5].copy()
df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
df_11 = df.loc[df['nummets'] >= 11].copy()

CPH_bootstrap(big1_top40_fp, sub=df_5)
CPH_bootstrap(big1_top40_fp, sub=df_5_10)
CPH_bootstrap(big1_top40_fp, sub=df_11)

df = pd.read_csv(big1_top40_fp, index_col=0)

# map labels to radiomic data
df['nummets'] = df.index.to_series().map(nummetsdic)
df['nummets'] = df['nummets'].astype(int)

# create sub-groups
df_5 = df.loc[df['nummets'] < 5].copy()
df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
df_11 = df.loc[df['nummets'] >= 11].copy()

CPH_bootstrap(big1_top40_fp, num=True, sub=df_5)
CPH_bootstrap(big1_top40_fp, num=True, sub=df_5_10)
CPH_bootstrap(big1_top40_fp, num=True, sub=df_11)

df = pd.read_csv(smallest_top40_fp, index_col=0)

# map labels to radiomic data
df['nummets'] = df.index.to_series().map(nummetsdic)
df['nummets'] = df['nummets'].astype(int)

# create sub-groups
df_5 = df.loc[df['nummets'] < 5].copy()
df_5_10 = df.loc[(df['nummets'] >= 5) & (df['nummets'] <= 10)].copy()
df_11 = df.loc[df['nummets'] >= 11].copy()

CPH_bootstrap(smallest_top40_fp, sub=df_5)
CPH_bootstrap(smallest_top40_fp, sub=df_5_10)
CPH_bootstrap(smallest_top40_fp, sub=df_11)

# -------------------- #
# Sub-Analysis: Volume Largest Met <200, 200-700, >700
# -------------------- #

df = pd.read_csv(UWA_top40_fp, index_col=0)

# map labels to radiomic data
df['volume'] = df.index.to_series().map(big1voldic)
df['volume'] = df['volume'].astype(int)

# create sub-groups
df_200 = df.loc[df['volume'] < 200].copy()
df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
df_700 = df.loc[df['volume'] >= 700].copy()

CPH_bootstrap(UWA_top40_fp, sub=df_200)
CPH_bootstrap(UWA_top40_fp, sub=df_200_700)
CPH_bootstrap(UWA_top40_fp, sub=df_700)

df = pd.read_csv(WA_top40_fp, index_col=0)

# map labels to radiomic data
df['volume'] = df.index.to_series().map(big1voldic)
df['volume'] = df['volume'].astype(int)

# create sub-groups
df_200 = df.loc[df['volume'] < 200].copy()
df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
df_700 = df.loc[df['volume'] >= 700].copy()

CPH_bootstrap(WA_top40_fp, sub=df_200)
CPH_bootstrap(WA_top40_fp, sub=df_200_700)
CPH_bootstrap(WA_top40_fp, sub=df_700)

df = pd.read_csv(big3_top40_fp, index_col=0)

# map labels to radiomic data
df['volume'] = df.index.to_series().map(big1voldic)
df['volume'] = df['volume'].astype(int)

# create sub-groups
df_200 = df.loc[df['volume'] < 200].copy()
df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
df_700 = df.loc[df['volume'] >= 700].copy()

CPH_bootstrap(big3_top40_fp, sub=df_200)
CPH_bootstrap(big3_top40_fp, sub=df_200_700)
CPH_bootstrap(big3_top40_fp, sub=df_700)

df = pd.read_csv(big1_top40_fp, index_col=0)

# map labels to radiomic data
df['volume'] = df.index.to_series().map(big1voldic)
df['volume'] = df['volume'].astype(int)

# create sub-groups
df_200 = df.loc[df['volume'] < 200].copy()
df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
df_700 = df.loc[df['volume'] >= 700].copy()

CPH_bootstrap(big1_top40_fp, sub=df_200)
CPH_bootstrap(big1_top40_fp, sub=df_200_700)
CPH_bootstrap(big1_top40_fp, sub=df_700)

df = pd.read_csv(big1_top40_fp, index_col=0)

# map labels to radiomic data
df['volume'] = df.index.to_series().map(big1voldic)
df['volume'] = df['volume'].astype(int)

# create sub-groups
df_200 = df.loc[df['volume'] < 200].copy()
df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
df_700 = df.loc[df['volume'] >= 700].copy()

CPH_bootstrap(big1_top40_fp, num=True, sub=df_200)
CPH_bootstrap(big1_top40_fp, num=True, sub=df_200_700)
CPH_bootstrap(big1_top40_fp, num=True, sub=df_700)

df = pd.read_csv(smallest_top40_fp, index_col=0)

# map labels to radiomic data
df['volume'] = df.index.to_series().map(big1voldic)
df['volume'] = df['volume'].astype(int)

# create sub-groups
df_200 = df.loc[df['volume'] < 200].copy()
df_200_700 = df.loc[(df['volume'] >= 200) & (df['volume'] <= 700)].copy()
df_700 = df.loc[df['volume'] >= 700].copy()

CPH_bootstrap(smallest_top40_fp, sub=df_200)
CPH_bootstrap(smallest_top40_fp, sub=df_200_700)
CPH_bootstrap(smallest_top40_fp, sub=df_700)



