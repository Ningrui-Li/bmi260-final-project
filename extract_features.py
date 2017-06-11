from os import listdir
from os.path import join

import numpy as np
from scipy.stats import moment
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mahotas

import dicom
import SimpleITK as sitk


from helper import *

def extract_metadata_features(finding):
    '''
    Given a directory with MR image slices (DICOM), look at the first DICOM
    file and extract various metadata features from the patient, like his age.
    '''
    # Loop through all image volumes and read their DICOM files for info about
    # the patient's age and weight.
    for _, item in finding.items():
        if type(item) is dict:
            mri_dir = item['filepath']
            dcm_names = listdir(mri_dir)
            num_images = len(dcm_names)

            # Read in first image to get patient's age and weight.
            dcm = dicom.read_file(join(mri_dir, dcm_names[0]))
            features = [int(dcm.PatientAge[:-1]), int(dcm.PatientWeight)]

            return features


def extract_dce_features(dce_dir, finding_pos):
    '''
    Extract features from the provided DCE image volume given the finding's
    approximate location (in image coordinates).
    '''
    # Get header file name.
    for f in listdir(dce_dir):
        if f.endswith('mhd'):
            header_file = f

    # Read image volume and swap the dimensions around such that the first two
    # dimensions are the transverse directions and the last one is axial.
    img_itk = sitk.ReadImage(join(dce_dir, header_file))
    img = sitk.GetArrayFromImage(img_itk)
    img = np.transpose(img, (2, 1, 0))

    # Convert finding coordinates to image volume indices.
    finding_idx = img_itk.TransformPhysicalPointToIndex(finding_pos)
    x_index = finding_idx[1]
    y_index = finding_idx[0]
    slice_index = img.shape[2]-finding_idx[2]-1

    '''
    roi = img[x_index-4:x_index+4,
              y_index-4:y_index+4,
              slice_index-1:slice_index+2]
    '''
    roi = img[x_index-2:x_index+3,
              y_index-2:y_index+3,
              slice_index-1:slice_index+2]

    '''
    # Add ROI dashed rectangle around the finding.
    roi_size = 14
    finding_roi = patches.Rectangle(
        (x_index-roi_size/2, y_index-roi_size/2),
        roi_size, roi_size, fill=False, color='b', linewidth=2,
        linestyle='dashed')
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, slice_index], cmap='gray')
    ax.add_patch(finding_roi)

    plt.show()
    '''

    # Histogram-based features.
    histogram_features = [np.max(roi), np.mean(roi), np.median(roi),
                          np.std(roi), np.max(img), np.mean(img),
                          np.median(img), np.std(img),
                          img[finding_idx[1], finding_idx[0], slice_index]]

    # Or just use the entire ROI as a feature!
    #dce_features = list(roi.flatten())

    return histogram_features


def extract_zone_features(zone):
    '''
    Perform one-hot encoding of the zone where the finding is located. Allows
    the zone to be used as a feature even though it's a categorical variable.
    '''
    if zone == 'PZ':
        return [1, 0, 0]
    elif zone == 'AS':
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def extract_t2_features(mri_dir, finding_idx):
    '''
    Extract features from the provided T2 image volume given the finding's
    approximate location (in image indices).
    '''
    img_vol = read_mri_volume(mri_dir)

    # Slice index actually indexes in reverse.
    slice_index = img_vol.shape[2]-finding_idx[2]-1
    roi = img_vol[finding_idx[1]-15:finding_idx[1]+15,
        finding_idx[0]-15:finding_idx[0]+15,
        slice_index-1:slice_index+2]

    '''
    # Show ROI for debugging.
    print(img_vol.shape, finding_idx)
    fig, ax = plt.subplots()
    ax.imshow(img_vol[finding_idx[1]-100:finding_idx[1]+100,
        finding_idx[0]-100:finding_idx[0]+100, slice_index, cmap='gray')
    ax.set_title('T2 ROI')
    plt.tight_layout()
    plt.show()
    '''

    # Histogram-based features.
    histogram_features = [np.max(roi), np.mean(roi),
        np.median(roi), np.std(roi)]

    # Compute texture features.
    texture_features = list(mahotas.features.haralick(roi,
        compute_14th_feature=False).flatten())

    # Or just use the entire ROI as a feature!
    #t2_features = list(roi.flatten())

    t2_features = histogram_features + texture_features

    return t2_features


def extract_adc_features(mri_dir, finding_idx):
    '''
    Extract features from the provided ADC map given the finding's
    approximate location (in image indices).
    '''
    img_vol = read_mri_volume(mri_dir)

    x_index = finding_idx[1]
    y_index = finding_idx[0]
    slice_index = img_vol.shape[2]-finding_idx[2]-1

    roi = img_vol[x_index-2:x_index+3, y_index-2:y_index+3,
        slice_index-1:slice_index+2]

    # Show ROI for debugging.
    '''
    print(img_vol.shape, finding_idx)
    fig, ax = plt.subplots()
    ax.imshow(img_vol[x_index-20:x_index+20,
        y_index-20:y_index+20, slice_index], cmap='gray')
    ax.set_title('ADC ROI')
    plt.tight_layout()
    plt.show()
    '''

    # Histogram-based features.
    histogram_features = [np.max(roi), np.mean(roi),
        np.median(roi), np.std(roi)]

    # Compute texture features.
    texture_features = list(mahotas.features.haralick(roi,
        compute_14th_feature=False).flatten())

    # Or just use the entire ROI as a feature!
    adc_features = list(roi.flatten())

    adc_features = histogram_features + texture_features
    
    return adc_features
