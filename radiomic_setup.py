# Pyradiomics Set-Up

"""
The input file for batch processing is a CSV file where the first row contains headers and each subsequent row
represents one combination of an image and a segmentation and contains at least 2 elements:
1) path/to/image, 2) path/to/mask. The headers specify the column names and must be “Image” and “Mask” for image and
mask location, respectively (capital sensitive). Additional columns may also be specified, all columns are copied to
the output in the same order (with calculated features appended after last column).

https://pyradiomics.readthedocs.io/en/latest/usage.html
"""

import pandas as pd
from Main.Core_Functions import listdir_nohidden as listdir_nohidden

machine = 'AWS'  # vs local

if machine is 'AWS':
	ROOT = '/radiomics/'
elif machine is 'local':
	ROOT = '/Users/enochchang/brainmets/'
else:
	ROOT = None

imagefolder = ROOT + 'images/'
maskfolder = ROOT + 'masks/'
imagefoldernohidden = listdir_nohidden(imagefolder)
patientimages = sorted(list(imagefoldernohidden))
IMAGEMASKFILE = ROOT + 'imagemaskfile.csv'

x = len(patientimages)
data = []

# extract row_ID and metnum (filename is N4ITKimagefolder + str(row_ID) + '_' + str(metcount) + '_image.nrrd')
# save image and mask file names to csv
for x in range(0, x):
	imagepath = 'images/' + str(patientimages[x])
	maskpath = 'masks/' + str(patientimages[x].split('image.nrrd')[0] + "mask.nrrd")
	row_ID, metnum = patientimages[x].split('_')[:2]
	data.append([imagepath, maskpath, row_ID, metnum])

df = pd.DataFrame(data, columns=['Image', 'Mask', 'row_ID', 'metnum'])
df.to_csv(IMAGEMASKFILE, index=False)

print('image mask csv file saved')
