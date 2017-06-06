from os import listdir
from os.path import join

import dicom

def extract_metadata_features(patient_mri_dir):
    '''
    Given a directory with MR image slices (DICOM), look at the first DICOM
    file and extract various metadata features from the patient, like his age.
    '''
    dcm_names = listdir(patient_mri_dir)
    num_images = len(dcm_names)

    # Read in first image to get image size info.
    dcm = dicom.read_file(join(patient_mri_dir, dcm_names[0]))

    # Add patient's age, size, and weight.
    features = [dcm.PatientAge[:-1], dcm.PatientSize, dcm.PatientWeight]

    # Cast everything to a float.
    features = [float(feature) for feature in features]

    return features
