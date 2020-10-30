# Link Clinical and Imaging Data
# By Aneja Lab - Enoch Chang

import numpy as np
import pandas as pd
import random

# -------------------- #
# folders
# -------------------- #

# clinical variable database containing cohort only (no prior resection or radiation)
CLINICAL = '/Users/enochchang/brainmets/cohort_clinical_data.xlsx'

# from image preprocessing
SAVED_METCOUNT = '/Users/enochchang/brainmets/df_radiomics.npy'

ROOT = '/Users/enochchang/brainmets/'

# file for linked image IDs + clincial data
COHORT_fp = ROOT + 'linked_cohort.csv'

# -------------------- #
# load clinical variables
# -------------------- #

# load clinical database
clin = pd.read_excel(CLINICAL)

# load data from image preprocesing
imagecounts = np.load(SAVED_METCOUNT, allow_pickle=True).item()

columns = ['Row_ID', 'metnum', 'Z', 'XY']
df = pd.DataFrame(columns=columns)
count = 0
for key in sorted(imagecounts):
    metlist = imagecounts.get(key)
    count += len(metlist)
    for i in range(len(metlist)):
        data = pd.DataFrame({'Row_ID': int(key),'metnum': int(metlist[i][0]), 'Z': int(metlist[i][1]),
                            'XY': int(metlist[i][2])}, index=[key])
        df = df.append(data, ignore_index=True)

# save labels mapped to IDs
survdic = pd.Series(clin['survival'],index=clin['Row_ID']).to_dict()
deaddic = pd.Series(clin['dead'].values,index=clin['Row_ID']).to_dict()
agedic = pd.Series(clin['Age'].values,index=clin['Row_ID']).to_dict()
kpsdic = pd.Series(clin['KPS'].values,index=clin['Row_ID']).to_dict()
ecmdic = pd.Series(clin['ECM'].values,index=clin['Row_ID']).to_dict()

df['survival'] = df['Row_ID'].map(survdic)
df['dead'] = df['Row_ID'].map(deaddic)
df['Age'] = df['Row_ID'].map(agedic)
df['KPS'] = df['Row_ID'].map(kpsdic)
df['ECM'] = df['Row_ID'].map(ecmdic)

# if clinical variable for number of mets is missing, fill with max value for met num per patient from preprocessing
# (this was recorded when iteratively extracting each met during preprocessing)
df.loc[df['#of Mets Treated'].isnull(), '#of Mets Treated'] = df.groupby('Row_ID')['metnum'].transform('max')
nummetsdic = pd.Series(df['#of Mets Treated'].values,index=df['Row_ID']).to_dict()
np.save(ROOT + 'nummetsdic.npy', nummetsdic)

# drop mets < 5 mm
df = df[df['XY'] >= 5]

# drop non-cohort patients
cohort = df.loc[(df['survival'].notnull())].copy()
cohort = cohort.astype({'dead':'int', 'survival':'int'})

# save linked image IDs + clincial data dataset
cohort.to_csv(COHORT_fp, index=True)


