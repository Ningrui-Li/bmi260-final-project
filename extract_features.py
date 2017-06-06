from os import listdir
from os.path import join

import dicom
import SimpleITK as sitk

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


def extract_dce_features(dce_dir):
    '''
    Extract features from the provided DCE image.
    '''
    # Get header and image volume file names.
    for f in listdir(dce_dir):
        if f.endswith('mhd'):
            header_file = f
        else:
            img_file = f

    img_itk = sitk.ReadImage(join(dce_dir, header_file))
    img = sitk.GetArrayFromImage(img_itk)
    print(img.shape)
    return
