from os import listdir
from os.path import join

import numpy as np

import dicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def extract_metadata_features(mri_dir):
    '''
    Given a directory with MR image slices (DICOM), look at the first DICOM
    file and extract various metadata features from the patient, like his age.
    '''
    dcm_names = listdir(mri_dir)
    num_images = len(dcm_names)

    # Read in first image to get image size info.
    dcm = dicom.read_file(join(mri_dir, dcm_names[0]))

    # Add patient's age, size, and weight.
    features = [int(dcm.PatientAge[:-1]), int(dcm.PatientWeight)]

    return features


def extract_dce_features(dce_dir, finding_pos):
    '''
    Extract features from the provided DCE image volume given the finding's
    approximate location (in image coordinates).
    '''
    # Get header and image volume file names.
    for f in listdir(dce_dir):
        if f.endswith('mhd'):
            header_file = f
        else:
            img_file = f

    # Read image volume and swap the dimensions around such that the first two
    # dimensions are the transverse directions and the last one is axial.
    img_itk = sitk.ReadImage(join(dce_dir, header_file))
    img = sitk.GetArrayFromImage(img_itk)
    img = np.transpose(img, (2, 1, 0))

    # Convert finding coordinates to image volume indices.
    idx_finding = img_itk.TransformPhysicalPointToIndex(finding_pos)
    print(idx_finding)

    # Add ROI dashed rectangle around the finding.
    roi_size = 14
    finding_roi = patches.Rectangle(
        (idx_finding[1]-roi_size/2, idx_finding[0]-roi_size/2),
        roi_size, roi_size, fill=False, color='b', linewidth=2,
        linestyle='dashed')
    fig, ax = plt.subplots()
    ax.imshow(img[:, :, idx_finding[2]], cmap='gray')
    ax.add_patch(finding_roi)
    ax.add_patch(finding_roi)
    plt.show()
    return
