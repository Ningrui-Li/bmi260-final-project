from os import listdir
from os.path import join

import numpy as np
from scipy.stats import moment
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    idx_finding = img_itk.TransformPhysicalPointToIndex(finding_pos)

    roi = img[idx_finding[1]-4:idx_finding[1]+4,
              idx_finding[0]-4:idx_finding[0]+4,
              idx_finding[2]-1:idx_finding[2]+2]

    '''
    # Add ROI dashed rectangle around the finding.
    roi_size = 14
    finding_roi = patches.Rectangle(
        (idx_finding[1]-roi_size/2, idx_finding[0]-roi_size/2),
        roi_size, roi_size, fill=False, color='b', linewidth=2,
        linestyle='dashed')
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, idx_finding[2]], cmap='gray')
    ax.add_patch(finding_roi)

    fig, ax = plt.subplots()
    ax.hist(roi.flatten(), 40)
    ax.set_xlim(0, 40)
    plt.show()
    '''

    # Histogram-based features.
    dce_features = [np.max(roi), np.mean(roi), np.median(roi), np.std(roi),
                    np.max(img), np.mean(img), np.median(img), np.std(img),
                    img[idx_finding[1], idx_finding[0], idx_finding[2]]]

    # Or just use the entire ROI as a feature!
    #dce_features = list(roi.flatten())

    return dce_features


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
    roi = img_vol[finding_idx[1]-2:finding_idx[1]+3,
        finding_idx[0]-2:finding_idx[0]+3,
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
    t2_features = [np.max(roi), np.mean(roi), np.std(roi)]

    # Or just use the entire ROI as a feature!
    #t2_features = list(roi.flatten())

    return t2_features


def extract_adc_features(mri_dir, finding_idx):
    '''
    Extract features from the provided ADC map given the finding's
    approximate location (in image indices).
    '''
    img_vol = read_mri_volume(mri_dir)
    roi = img_vol[finding_idx[0]-2:finding_idx[0]+3,
        finding_idx[1]-2:finding_idx[1]+3,
        finding_idx[2]-1:finding_idx[2]+2]

    # Histogram-based features.
    #adc_features = [np.max(roi), np.mean(roi), np.std(roi)]

    # Or just use the entire ROI as a feature!
    adc_features = list(roi.flatten())

    return adc_features
