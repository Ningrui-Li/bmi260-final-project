import csv
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import dicom


def read_finding_labels(findings_filename, images_filename, mri_dir, dce_dir):
    '''
    INPUT:
    findings_filename - name of CSV file with labels for each finding.
    images_filename - name of CSV file with info for each image acquisition.
    dce_dir - name of directory containing Ktrans images for each patient.s


    OUTPUTS:
    patients - dictionary that maps patient names to another dictionary
               containing information about each patient (like their name,
               label, filepaths to MR images, etc.)
    '''
    findings = dict()

    with open(findings_filename) as finding_labels:
        next(finding_labels) # Skip first line (because it's a header).

        for line in finding_labels:
            # Ignore blank lines.
            line = line.strip()
            if not line:
                continue

            line = line.split(',')

            finding = dict()
            finding['patient_name'] = line[0]
            finding['id'] = int(line[1])
            finding['pos'] = line[2]
            finding['zone'] = line[3]
            finding['score'] = int(line[4])

            # Set finding name to be the patient name followed by the
            # finding number.
            finding_name = '{}-{}'.format(finding['patient_name'],
                finding['id'])


            findings[finding_name] = finding

    with open(images_filename) as images_labels:
        csv_reader = csv.reader(images_labels, delimiter=',', quotechar='"')
        next(csv_reader)

        for line in csv_reader:
            if not line:
                continue

            patient_name = line[0]
            patient_mri_dir = get_mri_dir(mri_dir, patient_name)
            seq_name = line[1] # Name of MR pulse sequence.
            seq_id = line[-1] # DICOM ID of image series.
            #acquisition_date = line[2] # Date of scan.
            fid = int(line[3]) # Finding ID.
            idx = [int(i) for i in line[6].split(' ')] # Indices of findings.
            spacing = [float(i) for i in line[8].split(',')] # Voxel Spacing
            dim = [int(i) for i in line[9].split('x')]
            dim = dim[:-1]

            finding_name = '{}-{}'.format(patient_name, fid)

            # Get filepath to images acquired for a given pulse sequence.
            if seq_name not in findings[finding_name]:
                findings[finding_name][seq_name] = dict()

            findings[finding_name][seq_name]['filepath'] = join(
                patient_mri_dir, seq_id)
            findings[finding_name][seq_name]['world_matrix'] = line[5]
            findings[finding_name][seq_name]['finding_idx'] = idx
            findings[finding_name][seq_name]['vox_spacing'] = spacing

            # Get filepath to Ktrans map.
            findings[finding_name]['dce'] = dict()
            findings[finding_name]['dce']['filepath'] = join(
                get_dce_dir(dce_dir, patient_name))

    return findings


def get_mri_dir(mri_dir, patient_name):
    '''
    Returns the sub-directory of MR images for a given patient.

    INPUTS:
    mri_dir - directory of MR images for all patients.
    patient_name - name of patient (for example, ProstateX-0014).

    OUTPUT:
    patient_mri_dir = directory of MR image series for the given patient.
    '''
    patient_mri_dir = join(mri_dir, patient_name)
    patient_mri_dir = join(patient_mri_dir, listdir(patient_mri_dir)[0])
    return patient_mri_dir


def get_dce_dir(dce_dir, patient_name):
    '''
    Returns the sub-directory of Ktrans images for a given patient.

    INPUTS:
    dce_dir - directory of DCE images for all patients.
    patient_name - name of patient (for example, ProstateX-0014).

    OUTPUT:
    patient_dce_dir = directory of MR image series for the given patient.
    '''
    return join(dce_dir, patient_name)


def read_mri_volume(patient_mri_dir):
    '''
    Reads in a series of DCM images into a 3D numpy.ndarray. The DCM images are
    not in the correct axial ordering, so this function also sorts them.
    INPUTS:
    patient_mri_dir (string) - path of folder with patient's DCM images.
    OUTPUT:
    img_vol - 3D numpy array of CT scan. First dimension is coronal, second
        dimension is sagittal, third dimension is axial.
    '''
    dcm_names = listdir(patient_mri_dir)
    num_images = len(dcm_names)

    # Read in first image to get image size info.
    dcm = dicom.read_file(join(patient_mri_dir, dcm_names[0]))

    # Allocate memory for image volume.
    img_vol = np.zeros((dcm.Rows, dcm.Columns, num_images)).astype('int16')
    axial_pos = [0]*num_images;

    # Read all image slices into the volume.
    for dcm_name in dcm_names:
        dcm = dicom.read_file(join(patient_mri_dir, dcm_name))

        # Convert image pixel values to Hounsfield units.
        slice_index = dcm.InstanceNumber-1

        # Store image axial location
        axial_pos[slice_index] = dcm.SliceLocation

        # Store image into volume.
        img_vol[:,:,slice_index] = dcm.pixel_array


    # Print resolution.
    axial_diff = [axial_pos[n]-axial_pos[n-1] for n in range(1, len(axial_pos))]
    axial_res = np.abs(np.mean(axial_diff))
    #print('The transverse resolution is {} mm.'.format(dcm.PixelSpacing))
    #print('The axial resolution is {} mm.'.format(axial_res))

    return img_vol


def optimize_svc_params(X_train, y_train, param_grid):
    '''
    Optimizes hyperparameters for SVC model based on provided training data
    using 10-fold cross validation.
    Only looks at the parameters given in 'param_grid'.
    '''
    clf = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid, cv=10)
    clf.fit(X_train, y_train)
    print("Optimal params:")
    print(clf.best_params_)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.04f) for %r"  % (mean, std, params))

    return clf.best_params_
