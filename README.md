# Comparison of Radiomic Feature Aggregation Methods for Patients with Multiple Tumors 

# Workflow

1. Run `preprocess.py` to isolate individual tumors from DICOM files with corresponding segmentations, correct for low 
frequency intensity non-uniformity present in MRI data with the N4ITK bias field correction algorithm, and z-score 
normalize to reduce inter-scan bias. This saves individual images and masks as `.nrrd` files in addition to a dictionary 
linking anonymous patient identifiers to tumor-level imaging metrics.
2. Setup a `.csv` file for input to the pyradiomics extraction pipeline using `radiomic_setup.py`.
3. Extract radiomic features using pyradiomics via the command line interface:
    ````
    pyradiomics imagemaskfile.csv -o radiomicsresults.csv -f csv --jobs 16 --param params.yaml
    ````
4. Link clinical variables to imaging data with `link_clinical_imaging.py`.
5. Compute various radiomic feature aggregation methods with `radiomic_aggregation.py`.
6. Dimensionality reduction of radiomic features via minimum redundancy maximum relevance using `mRMRe_selection.R`.
7. Load and format radiomic features selected via mRMRe with `selected_radiomic_loading.py`.
8. Train models using `survival_models.py`.

# Example Pre-Processing Images
1. Input: Slices of brain MRI scans loaded from DICOM files
2. Identification of region of interest: Tumor segmentation
3. Output: Extracted tumor of interest

<img width="711" alt="picture 2" src="https://user-images.githubusercontent.com/51829815/97620085-6fe7a280-19f7-11eb-8a9c-a46526bddde3.png">
 

