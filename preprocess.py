# RT Structure Preprocessing
# By Aneja Lab - Enoch Chang
# Modularization with Hannah Chang

# Import
import os
import pydicom
import numpy as np
from skimage.color import gray2rgb
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import scipy.ndimage
import cv2
from Main.Core_Functions import listdir_nohidden as listdir_nohidden
import h5py
import SimpleITK as sitk


"""
Folder Structure: 
	Patient 1
		IMG01.dcm
		IMG02.dcm
		RTSS.dcm
		RTDOSE.dcm
		RTPLAN.dcm
	Patient 2
		IMG01.dcm
		IMG02.dcm
		RTSS.dcm
		RTDOSE.dcm
		RTPLAN.dcm
"""

TRAINTEST = 'train' # (str): 'train' or 'test'
machine = 'AWS' # (str): 'AWS' or 'local'
stem = 'radiomics' # (str): identify sample

# ----------------#
# For AWS
# ----------------#
if machine == 'AWS':
    INPUT_FOLDER = '/data/Scans/' + TRAINTEST
    SAVED_HD5STACK = '/Constants/' + TRAINTEST + '2Dstack_' + stem + '.hdf5'
    SAVED_METCOUNT = '/Constants/' + TRAINTEST + 'metcounts_' + stem + '.npy'

    # Folder for saving individual structures as .nrrd files for radiomic feature extraction
    N4ITKimagefolder = '/radiomics/' + TRAINTEST + '/images/'
    N4ITKmaskfolder = '/radiomics/' + TRAINTEST + '/masks/'

    # Folder for saving individual images
    imagefolder = '/Constants/' + TRAINTEST + 'images/' + stem + "/"

# ----------------#
# For Local
# ----------------#
# FOR SAMPLE SET
if machine == 'local':
    INPUT_FOLDER = '/Users/enochchang/brainmets/Scans/Figures'
    SAVED_HD5STACK = '/Users/enochchang/brainmets/sample_stack.hd5'
    SAVED_METCOUNT = '/Users/enochchang/brainmets/samplemetcounts.npy'

    # Folder for saving individual structures as .nrrd files for radiomic feature extraction
    N4ITKimagefolder = '/Users/enochchang/brainmets/radiomics/images/'
    N4ITKmaskfolder = '/Users/enochchang/brainmets/radiomics/masks/'

    # For saving image dict (mapped to row_ID)
    SAVED_DICT = '/Users/enochchang/brainmets/sampledict.npy'

    # For saving image list
    SAVEDLIST_IMAGEONLY = '/Users/enochchang/brainmets/sampleimagelist.npy'
    SAVED_HDF5_IMAGELIST = '/Users/enochchang/brainmets/hdf5sampleimagelist.hd5'

    # Folder for saving sample images as .png
    SAVED_IMAGES = '/Users/enochchang/brainmets/Figures/'

    # Folder for saving individual images as .npy
    imagefolder = '/Users/enochchang/brainmets/' + TRAINTEST + 'images/'

# ----------------#
# Functions
# ----------------#

def DICOM_modality(dicom_path):
    """
    Separate files by modality.

    :param dicom_path: (string) directory for individual patient (assumes each patient is in a separate folder)
    :return: (tuple) lists of files grouped by modality
    """
    dicom_files = listdir_nohidden(dicom_path)
    RT_files = []
    MR_files = []
    RS_files = []
    RP_files = []
    RD_files = []
    unclassified = []
    for i in dicom_files:
        dicom_type = pydicom.read_file(dicom_path + '/' + i)
        if dicom_type.Modality == 'RTSTRUCT':
            RS_files.append(i)
        elif dicom_type.Modality == 'MR':
            MR_files.append(i)
        elif dicom_type.Modality == 'RTRECORD':
            RT_files.append(i)
        elif dicom_type.Modality == 'RTPLAN':
            RP_files.append(i)
        elif dicom_type.Modality == 'RTDOSE':
            RD_files.append(i)
        else:
            unclassified.append(i)
    print('RT Treatment Files:', len(RT_files))
    print('MR Image Files:', len(MR_files))
    print('Structure Files:', len(RS_files))
    print('RT Plan Files:', len(RP_files))
    print('RT Dose Files:', len(RD_files))
    print('Unclassified Files (please verify):', len(unclassified))
    return RT_files, MR_files, RS_files, RP_files, RD_files, unclassified


def load_scan(path, MR_files):
    """
    Load scan in given folder path and determine slice thickness.

    :param path: (string) directory for individual patient (assumes each patient is in a separate folder)
    :param MR_files: (list) MR dicom files (unsorted)
    :return: (list) MR files sorted by z-position
    """
    slices = [pydicom.read_file(path + '/' + a) for a in MR_files]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    # Calculate pixel size in Z direction and add to metadata
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def get_pixels(slices):
    """
    Extract pixel values from dicom files.

    :param slices: (list) MR files sorted by z-position
    :return: (array) stack of pixels from all slices in float32
    """
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.float32)
    return np.array(image, dtype=np.float32)


def get_bbox(x1_box, x2_box, y1_box, y2_box, include_normal=False, max_slice_size=None):
    """
    Extract bounding box image coordinates of target structure.
    Bounding box is always a square with dimensions the max(max_slice_size, delta) centered around structure.

    :param max_slice_size: (int) max dimension of bounding box in XY plane; if None, no normal tissue padding
    :param x1_box: (float) min x image coordinate
    :param x2_box: (float) max x image coordinate
    :param y1_box: (float) min y image coordinate
    :param y2_box: (float) max y image coordinate
    :param include_normal: (bool) if True, pad bounding box with normal tissue up to 'max_slice_size';
        otherwise, bounding box does not include padding. Include normal tissue up to 'max_slice_size' around the center
        of structure as long as XY dimensions (delta) of structure are < 'max_slice_size'.
    :return: bbox_coord (array) bounding box image coordinates as int
             delta (float) maximum pixel dimension of structure in XY plane
    """
    # calculate delta
    if abs((x1_box - x2_box)) > abs((y1_box - y2_box)):
        delta = abs((x1_box - x2_box))
    else:
        delta = abs((y1_box - y2_box))

    # pad bounding box with normal tissue up to max_slice_size
    if include_normal and max_slice_size != None and delta < max_slice_size:
        y_center = round((y1_box + y2_box) / 2)
        x_center = round((x1_box + x2_box) / 2)

        bbox_coord = np.asarray(
            ((x_center - np.floor(max_slice_size / 2)), (x_center + np.ceil(max_slice_size / 2))),
            ((y_center - np.floor(max_slice_size / 2)), (y_center + np.ceil(max_slice_size / 2))))

    # bbox coordinates without padding
    else:
        bbox_coord = np.asarray([(x1_box, (x1_box + delta)), (y1_box, (y1_box + delta))])

    return bbox_coord, delta


def resample(image, scan, new_spacing=None):
    """
    Resample scans to adjust for differences in slice thickness and resolution.

    :param image: (array) image array
    :param scan: (list) MR files sorted by z-position
    :param new_spacing: (list) new spacing [mm, mm, mm]
    :return: (array) resampled image array
    """
    if new_spacing is None:
        new_spacing = [1, 1, 1]

    spacing_list = [scan[0].SliceThickness]
    spacing_list.extend(scan[0].PixelSpacing)
    spacing = np.array(spacing_list, dtype=np.float32)

    # define variable resize factor using current spacing divided by new spacing given above
    resize_factor = spacing / new_spacing

    # dimension of the image array * resize factor
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape

    # resample images with new resize factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    return image


def format_image(image, xy_size=224, z_size=None, zero_norm_slicelvl=False):
    """
    Resize image in XY plane to xy_size.
    Pad to z_size in z direction with black if specified; else remove top and bottom slice if total slices > 3.
    Zero norm at slice level if specified.

    :param image: (array) image array
    :param xy_size: (int) resize XY plane to this size (Resnet format is 224 x 224)
    :param z_size: (int) if specified, pad slices in z direction to z_slice
    :param zero_norm_slicelvl: (bool) if True, zero norm at slice level
    :return: (array) formatted array
    """
    num_slices = image.shape[0]

    # original image will be resized to an array of this dimension in XY plane
    tmp_image = np.zeros((num_slices, xy_size, xy_size))

    # resize every slice in original image to xy_size
    for idx in range(num_slices):
        img = image[idx, :, :]
        img_resized = cv2.resize(img, dsize=(xy_size, xy_size), interpolation=cv2.INTER_LINEAR)

        # zero norm at the slice level
        if zero_norm_slicelvl:
            tmp_image[idx, :, :] = zero_normalize(img_resized)
        else:
            tmp_image[idx, :, :] = img_resized


    # do not pad in z direction if z_size is not specified
    if z_size is None:
        # keep all slices if number of slices < 4; otherwise, exclude top and bottom slices
        if num_slices >= 4:
            tmp_image = tmp_image[1:-1, :, :]
            num_slices = num_slices - 2
        formatted_image = tmp_image

    # pad up to z_size or take the middle z_size slices
    else:
        formatted_image = np.zeros((z_size, xy_size, xy_size))
        if num_slices <= z_size:
            # pad in z-direction at end of resized image
            formatted_image[:num_slices, :, :] = tmp_image
        else:
            # slice the middle z_size slices of the original image (there will be no padding)
            z_center = int(np.floor(tmp_image.shape[0] / 2))
            z_min = int(z_center - (z_size / 2))
            z_max = int(z_center + (z_size / 2))
            formatted_image = tmp_image[z_min:z_max, :, :]

    print('formatted image shape:', formatted_image.shape)

    return formatted_image

def N4BiasCorrection(image):
    """
    N4 Bias Correction
    :param image: (array)
    :return: (array)
    """

    inputImage = sitk.GetImageFromArray(image)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

    # maskImage: specify which pixels used to estimate bias-field and suppress pixels close to zero
    # given no mask, OtsuThreshold converts grayscale to black/white; same exact shape as inputImage
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output = corrector.Execute(inputImage, maskImage)

    return sitk.GetArrayFromImage(output)


def zero_normalize(image):
    """
    Zero-center and standardize image pixel values.

    :param image: (array) either one slice, met, or stack
    :return: (array) zero-centered and standardized image
    """
    return (image - np.mean(image)) / np.std(image)


def RT_structure_isolate(path,
                         MR_files,
                         RS_files,
                         structure=None,
                         include_old=False,
                         include_normal=False,
                         max_slice_size=None,
                         minXY=0,
                         RESAMPLE=False,
                         CENTER=False,
                         grayimage=False,
                         N4_Bias=False,
                         N4ITKimagefolder=None,
                         save_each_image=False,
                         row_ID=None,
                         padvalue=0,
                         xy_size=224,
                         z_size=None,
                         zero_norm_slicelvl=False,
                         debugZ=False,
                         stack_2D=None,
                         saveimagelist=None,
                         forradiomics=False):
    """
    For one patient: identify specified structure contours and determine bounding box from contour coordinates.
    Extract formatted image arrays of each structure based on corresponding bounding box coordinates.

    :param path: (string) directory of patient folders with dicom files inside
    :param MR_files: (list) MR dicom files (unsorted)
    :param RS_files: (list) dicom file containing contour info
    :param structure: (string) name of structure to include
        e.g. keep: Met, Met_A, Left Frontal Met, Right Parietal Met #1, Lt Frontal Met, Rt Frontal Met
        e.g. exclude: Skull, Plan1_TgtA_20.0Gy@50%
    :param include_old: (bool) if True, include previously treated structures (those with 4 digits in name)
        e.g. "Met" and "Met_0809"
    :param include_normal: (bool) if True, pad bounding box with normal tissue up to 'max_slice_size';
        otherwise, bounding box does not include padding. Include normal tissue up to 'max_slice_size' around the center
        of structure as long as XY dimensions (delta) of structure are < 'max_slice_size'.
    :param max_slice_size: (int) max dimension of bounding box in XY plane; if None, no normal tissue padding
    :param minXY: (int) min XY dimension
    :param RESAMPLE: (bool) if True, resample
    :param CENTER: (bool) if True, extract only center slice
    :param grayimage: (bool) if True, extract grayscale image (1 channel); else rgb (3 channel)
    :param N4_Bias: (bool) if True, implement N4 Bias Correction
    :param N4ITKimagefolder: (str) folder containing individual formatted ITK images in .nrrd
    :param save_each_image: (bool) if True, save individual images in imagefolder as .npy
    :param row_ID: (byte) unique ID
    :param padvalue: (int) amount of XY padding of normal brain to include top/bottom and left/right in original image
    :param xy_size: (int) resize XY plane to this size (Resnet format is 224 x 224)
    :param z_size: (int) if specified, pad slices in z direction to z_slice
    :param zero_norm_slicelvl: (bool) if True, zero norm at slice level; else zero norm at entire met level
    :param debugZ: (bool) identify bugs (Z grid values out of bounds, etc)
    :param stack_2D: (bool) if True, stack all 2D images for all structures for all patients into one array
    :param saveimagelist: (bool) if True, save image only list (no metcount)
    :param forradiomics: (bool) if True, format image without resizing to uniform Resnet dimensions
    :return: (list) if stack_2D: list of images only [image1, image2, image3];
        else list of lists <[met number, image array]> [[1, image1], [2, image2], [3, image3]...]
    """
    if structure is None:
        print('RT Structure Is Not Specified!')
        structureName = input('Please Enter RT Structure Name')
    else:
        structureName = structure
    if len(RS_files) > 1:
        print("NOTE: Multiple RS Files Found. Only 1st File Will Be Parsed.")

    structures = pydicom.read_file(path + '/' + RS_files[0])

    metcount = 0
    notmetcount = 0
    temp_list = []  # for storing list of all mets for this patient: <[[met #1 (this patient's), image array #1], ...]>
    temp_counting_list = [] # for storing list [metcount, num z slices, xy dimension] for each met
    debugstate = False

    # define pattern (4 digits) for regex
    pattern = re.compile("\d{4}")

    # process each structure in the contour file
    for n, item in enumerate(structures.StructureSetROISequence):
        print(item.ROIName)
        structureNum = n
        # search for 4 digits in structure name (previously treated mets have 4 digit date)
        m = re.search(pattern, str(item.ROIName))

        if ((str(structureName).lower() in (str(item.ROIName)).lower()) or any(
                x in str(item.ROIName) for x in ["Left", "Right", "Lt", "Rt"])) and (not m or include_old):
            # number of contour slices for each structure in the z direction
            numContours = len(structures.ROIContourSequence[int(structureNum)].ContourSequence)
            print("Number of contours:", numContours)

            contourZCoords = np.zeros(numContours)
            numContourPoints = np.zeros(numContours)
            ravel_coords = []

            # for each contour slice of structure
            for i in range(0, numContours):
                # ContourData is a sequence of (x,y,z) coordinates for every contour point in a closed plane,
                # eg x1,y1,z1,x2,y2,z2, so dividing by 3 gives the distinct number of individual contour points
                contour = np.array(structures.ROIContourSequence[int(structureNum)].ContourSequence[i].ContourData)
                numContourPoints[i] = contour.size // 3.0
                # every 3rd value in ContourData is the Z coordinate for that contour slice's plane
                contourZCoords[i] = contour[2]
                # add contour coordinates from this slice to cumulative aggregate of data for all previous slices
                ravel_coords = np.append(ravel_coords, contour, axis=None)

            # reshape aggregated array of contour coordinates shape [numContourPoints * 3, 1] into shape [numContourPoints,3]
            structureCoords = np.reshape(ravel_coords, (ravel_coords.size // 3, 3))

            x_y_z_Coords = structureCoords.T
            contourZCoordsSorted = np.sort(contourZCoords, kind='mergesort')
            xmax = np.amax(x_y_z_Coords[0])
            xmin = np.amin(x_y_z_Coords[0])
            ymax = np.amax(x_y_z_Coords[1])
            ymin = np.amin(x_y_z_Coords[1])
            zmax = contourZCoordsSorted[-1]
            zmin = contourZCoordsSorted[0]

            # load scan
            slices = load_scan(path, MR_files)
            slice_thickness = slices[0].SliceThickness
            # print('Slice Thickness:', slice_thickness)

            # extract x, y, z coordinates of the structure's upper left hand corner
            slice_info = slices[0]
            x_origin, y_origin, z_origin = slice_info.ImagePositionPatient
            # print('z origin', z_origin)

            # extract pixel values from dicom files
            image = get_pixels(slices)

            # calculate min/max x, y, z image coordinates
            Z1_grid = np.floor((zmin - z_origin) / slice_thickness)
            Z2_grid = np.ceil((zmax - z_origin) / slice_thickness)
            x1_box = np.floor((xmin - x_origin) / (slice_info.PixelSpacing[0]))
            x2_box = np.ceil((xmax - x_origin) / (slice_info.PixelSpacing[0]))
            y1_box = np.floor((ymin - y_origin) / (slice_info.PixelSpacing[1]))
            y2_box = np.ceil((ymax - y_origin) / (slice_info.PixelSpacing[1]))

            # debug
            if debugZ:
                if int(Z1_grid) > len(MR_files):
                    print("BUG FOUND!")
                    debugstate = True
                elif int(Z2_grid) > len(MR_files):
                    print("BUG FOUND!")
                    debugstate = True
                elif Z1_grid < 0:
                    print("BUG FOUND!")
                    debugstate = True
                elif Z2_grid < 0:
                    print("BUG FOUND!")
                    debugstate = True
                elif Z1_grid == Z2_grid:
                    print("BUG FOUND!")
                    debugstate = True

            # calculate bounding box coordinates and delta from contours
            bbox_coord, delta = get_bbox(x1_box, x2_box, y1_box, y2_box, include_normal, max_slice_size)

            # if delta is < minXY, skip the rest of preprocessing and continue to the next structure
            if delta < minXY:
                print('Met is less than', minXY, 'mm')
                print('_____')
                continue

            metcount += 1  # update metcount to record number of met actually processed for this patient

            # extract image pixel values (z, y, x) using bbox coordinates
            # include normal tissue of padvalue on each side of XY plane
            met_image = image[int(Z1_grid):int(Z2_grid),
                        (int(bbox_coord[1, 0]) - padvalue):(int(bbox_coord[1, 1]) + padvalue),
                        (int(bbox_coord[0, 0]) - padvalue):(int(bbox_coord[0, 1]) + padvalue)]
            print('Image shape before processing:', met_image.shape)

            # calculate pad offsets (floor both offsets so that odd numbers can still be in range)
            padoffset1 = int(np.floor((max_slice_size - delta) / 2))
            padoffset2 = int(np.floor((max_slice_size + delta) / 2 + (2 * padvalue)))

            # add black padding around center of structure up to max_slice_size if delta < max_slice_size
            # do not add black padding if for radiomics
            if delta > max_slice_size or forradiomics:
                final_image = met_image
                print('For radiomics')
            else:
                # image centered in padding (note: extra computation required above to center padding)
                final_image = np.zeros(
                    (met_image.shape[0], max_slice_size + (padvalue * 2), max_slice_size + (padvalue * 2)))
                final_image[:, padoffset1:padoffset2, padoffset1:padoffset2] = met_image

            # resample image
            if RESAMPLE:
                final_image = resample(final_image, slices)

            # -------------------------------------------------#
            # N4 Bias Field Correction (default settings)
            # -------------------------------------------------#
            if N4_Bias is True:
                final_image = N4BiasCorrection(final_image)

            # keep only center slice
            if CENTER:
                center_slice_idx = int(np.floor(final_image.shape[0] / 2))
                final_image = final_image[center_slice_idx:center_slice_idx + 1, :, :]

            # resize image to desired XY and Z dimensions (do not resize if for radiomics)
            if forradiomics:
                pass
            else:
                final_image = format_image(final_image, xy_size=xy_size, z_size=z_size, zero_norm_slicelvl=zero_norm_slicelvl)

            if grayimage:
                pass
            else:
                final_image = gray2rgb(final_image)

            # Resnet type is float32
            final_image = final_image.astype(np.float32)

            # zero-normalize here if norming over the entire met level
            if zero_norm_slicelvl:
                # print('Zero normalizing at the slice level')
                pass
            else:
                # print('Zero normalizing at the Met Level')
                final_image = zero_normalize(final_image)

            print('Final image shape:', final_image.shape)

            # save individual met images in separate files (.npy format)
            if save_each_image:
                print('Saving individual processed image')
                # save image in array format .npy
                np.save(imagefolder + str(row_ID) + '_' + str(metcount) + '_image.npy', final_image)

            # save individual met images and corresponding masks in separate files (.nrrd format)
            if N4ITKimagefolder:
                print('Saving individual processed image')
                # save image in ITK .nrrd format
                output = sitk.GetImageFromArray(final_image)
                sitk.WriteImage(output, N4ITKimagefolder + str(row_ID) + '_' + str(metcount) + '_image.nrrd')

                # create mask for pyradiomics
                ma_arr = np.ones((final_image.shape))
                ma = sitk.GetImageFromArray(ma_arr)
                ma = sitk.Cast(ma, sitk.sitkUInt32)

                # copy geometric information to align image and mask
                ma.CopyInformation(output)

                sitk.WriteImage(ma, N4ITKmaskfolder + str(row_ID) + '_' + str(metcount) + '_mask.nrrd')

            temp_list.append(final_image)

            # ----------------------------------------------------------- #
            # save list [metcount, num z slices, xy dimension] for each met
            # ----------------------------------------------------------- #
            temp_counting_list.append([metcount, final_image.shape[0], delta])

        else:
            notmetcount += 1

        print("metcount:", metcount)
        print("notmetcount:", notmetcount)
        print("_____")

    if metcount == 0:
        print('No mets extracted!')

    return temp_list, temp_counting_list, debugstate

def final_RT_image_preprocess(path,
                              N4ITKimagefolder=None,
                              include_old=False,
                              include_normal=False,
                              max_slice_size=None,
                              minXY=0,
                              xy_size=224,
                              z_size=None,
                              RESAMPLE=False,
                              CENTER=False,
                              grayimage=False,
                              N4_Bias=False,
                              padvalue=0,
                              debugZ=False,
                              zero_norm_slicelvl=False,
                              stack_2D=None,
                              saveimages=False,
                              saveimagelist=False,
                              save_each_image=False,
                              forradiomics=False
                              ):
    """
    Final preprocessing function.
    For all patients: extract formatted structure images with option to save 2D stack to hdf5 and save selection of
        sample images to png.

    :param path: (string) directory of patient folders with dicom files inside
    :param include_old: (bool) if True, include previously treated structures (those with 4 digits in name)
        e.g. "Met" and "Met_0809"
    :param include_normal: (bool) if True, pad bounding box with normal tissue up to 'max_slice_size';
        otherwise, bounding box does not include padding. Include normal tissue up to 'max_slice_size' around the center
        of structure as long as XY dimensions (delta) of structure are < 'max_slice_size'
    :param max_slice_size: (int) max dimension of bounding box in XY plane; if None, no normal tissue padding
    :param minXY: (int) min XY dimension
    :param z_size: (int) if specified, pad slices in z direction to z_slice
    :param RESAMPLE: (bool) if True, resample
    :param CENTER: (bool) if True, extract only center slice
    :param grayimage: (bool) if True, extract grayscale image (1 channel); else rgb (3 channel)
    :param N4_Bias: (bool) if True, implement N4 Bias Correction
    :param padvalue: (int) amount of XY padding of normal brain to include top/bottom and left/right in original image
    :param debugZ: (bool) identify bugs (Z grid values out of bounds, etc)
    :param zero_norm_slicelvl:
    :param stack_2D: (bool) if True, stack all 2D slices together; else keep images separated by patient
    :param saveimages: (bool) if True, save png images
    :param saveimagelist: (bool) #if True, save image only to list
    :param save_each_image: (bool) if True, save individual images in imagefolder
    :param forradiomics: (bool) if True, format image without resizing to uniform Resnet dimensions
    :return: (dictionary) if not stack_2D, return dictionary with list of [image] mapped to ID
        e.g. [{ID1:[image1, image2]}, {ID2:[image3]}]
    """

    patients_nohidden = listdir_nohidden(path)

    # sort systematically by row_id, just as the dict for the labels will also be sorted
    patients = sorted(list(patients_nohidden), key=lambda p: p.split('_')[0])
    x = len(patients)
    print('total patients', x)

    mets_processed_counter = 0
    RT_image_preprocess_list = []
    RT_image_preprocess_dict = {}
    metcount_dic = {}
    z_slice_counter = 0

    # debug
    debuglist = []
    debugnometpts = 0

    # extract formatted structure images for each patient
    for i in tqdm(range(0, x)):
        print("_________________________________")
        print("Currently processing patient", (i + 1))
        dicom_files = DICOM_modality(path + '/' + patients[i])
        patient_path = path + '/' + patients[i]
        print("patient path", patient_path)
        RT_files, MR_files, RS_files, RP_files, RD_files, unclassified = dicom_files

        # only process patients with MR files (exclude CT scans)
        if len(MR_files) == 0:
            debugnometpts += 1
            print('NOTE! No MRI Files')
        else:
            # extract unique patient treatment ID: patients[i] is in the format of "<rowID>_<MRN>"
            row_ID = patients[i].split('_')[0]

            new_images, counting_list, debugstate = RT_structure_isolate(patient_path, MR_files, RS_files, structure="met",
                                                          include_old=include_old, include_normal=include_normal,
                                                          max_slice_size=max_slice_size, minXY=minXY,
                                                          RESAMPLE=RESAMPLE, CENTER=CENTER, grayimage=grayimage,
                                                          N4_Bias=N4_Bias, N4ITKimagefolder=N4ITKimagefolder, row_ID=row_ID, padvalue=padvalue,
                                                          xy_size=xy_size, z_size=z_size, debugZ=debugZ,
                                                          zero_norm_slicelvl=zero_norm_slicelvl, stack_2D=stack_2D, saveimagelist=saveimagelist,
                                                          save_each_image=save_each_image, forradiomics=forradiomics)

            # debug
            if debugstate:
                print("ERROR BUG FOUND!")
                debuglist.append(row_ID)
            if debugZ:
                if len(new_images) == 0:
                    debugnometpts += 1
                    print('Debug! No mets!')

            # if stacking or saving images only to list, add processed images to list
            if stack_2D or saveimagelist:
                RT_image_preprocess_list.extend(new_images)
            # else add all structures for this patient to a dictionary mapped to row_ID
            else:
                RT_image_preprocess_dict.update({row_ID: new_images})

            # update cumulative counter
            mets_processed_counter += len(new_images)

            # update dict with structure counts, z slice num, xy size for each met
            metcount_dic.update({row_ID: counting_list})

            # update total z slices
            for a in range(len(counting_list)):
                z_slice_counter += counting_list[a][1]

            print("Total mets preprocessed:", mets_processed_counter)
            print("_________________________________")

    if debugZ:
        print('Total patients with no mets extracted:', debugnometpts)

    np.save(SAVED_METCOUNT, metcount_dic)
    print('Metcounts Saved')
    print("Total Z slices is:", z_slice_counter)


    # saving list of preprocessed mets (no metcount, format is [met1, met2, met3,...] )
    # note: the SAVED_METCOUNT file will have corresponding indices for matching row_id and labels to these mets
    if saveimagelist is True:
        print('Saving list of processed individual met images...')
        np.save(SAVEDLIST_IMAGEONLY, RT_image_preprocess_list)

    if stack_2D:
        HD5STACKZ = z_slice_counter  # for defining dataset shape
        print("Creating 2D Stack...")
        z_prev = 0

        # use hdf5 file format
        with h5py.File(SAVED_HD5STACK, 'w') as f:
            # create empty dataset with shape (total z slices, x, y, [rgb])
            if grayimage:
                dset = f.create_dataset('default', (HD5STACKZ, xy_size, xy_size), dtype=np.float32, compression=None)
            else:
                dset = f.create_dataset('default', (HD5STACKZ, xy_size, xy_size, 3), dtype=np.float32, compression=None)

            # iterate over list containing image arrays and add each structure to dataset stack
            for a in tqdm(range(0, len(RT_image_preprocess_list))):
                # determine number of slices and update z_offset accordingly
                tmp_z = RT_image_preprocess_list[a].shape[0]
                z_offset = z_prev + tmp_z
                # add structure to dataset
                dset[z_prev:z_offset, :, :] = RT_image_preprocess_list[a]
                # update z_prev
                z_prev = z_offset

            # save every 3rd slice as png
            if saveimages:
                print('Saving sample set of images...')
                for slice in tqdm(range(len(dset))):
                    if slice % 3 == 0: # TODO: customize how many images to sample
                        if grayimage:
                            plt.imsave(SAVED_IMAGES + str(slice) + '.png', dset[slice], cmap=plt.cm.gray)
                        else: # this is for RGB images
                            plt.imsave(SAVED_IMAGES + str(slice) + '.png', dset[slice].astype(np.uint8))

            print('Final', TRAINTEST, 'stack shape is', dset.shape)

        return (print('Preprocess Complete'))

    else:
        if debugZ:
            return RT_image_preprocess_dict, debuglist
        else:
            # np.save(SAVED_DICT, RT_image_preprocess_dict)
            # print("Dictionary Saved")
            print("Preprocess Complete")
            return RT_image_preprocess_dict

#######################################################################################################################
#######################################################################################################################

# For Radiomics (resample [1,1,1]), N4 bias correction, full met, grayscale 1 channel, no minimum size
output = final_RT_image_preprocess(INPUT_FOLDER, N4ITKimagefolder=N4ITKimagefolder, stack_2D=False,
                                           RESAMPLE=True, CENTER=False, grayimage=True, N4_Bias=True, padvalue=1,
                                           saveimages=False, minXY=0, zero_norm_slicelvl=False, saveimagelist=False,
                                              forradiomics=True)