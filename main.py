import csv
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import dicom
import numpy as np

from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

from extract_features import extract_metadata_features, extract_dce_features

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


def compute_finding_statistics(findings):
    finding_grades = [0, 0, 0, 0, 0]
    for _, finding in findings.items():
        finding_grades[int(finding['score'])-1] += 1
    print(finding_grades)
    return


def main():
    mri_dir = 'DOI'
    dce_dir = 'KtransTrain'
    findings_filename = 'ProstateX-2-Findings-Train.csv'
    images_filename = 'ProstateX-2-Images-Train.csv'

    #print(listdir(get_mri_dir(mri_dir, 'ProstateX-0014')))
    #print(listdir(get_dce_dir(dce_dir, 'ProstateX-0014')))
    #return

    findings = read_finding_labels(findings_filename, images_filename,
        mri_dir, dce_dir)
    compute_finding_statistics(findings)

    for _, fid in findings.items():
        # Convert finding score to a binary value.
        fid['score'] = int(fid['score'])
        if fid['score'] <= 1:
            fid['score'] = 0
        else:
            fid['score'] = 1
        

        # Convert finding position from string to coordinates.
        finding_pos = fid['pos'].split()
        finding_pos = [float(x) for x in finding_pos]

        metadata_features = []

        for fid_info in fid:
            #print(fid_info)
            # Process DCE images.
            if fid_info == 'dce':
                dce_features = extract_dce_features(
                    fid[fid_info]['filepath'], finding_pos)
            # Process transverse T2-weighted images and
            # extract metadata features.
            elif 't2_tse_tra0' in fid_info:
                idx = fid[fid_info]['finding_idx']
                #print('Finding at', idx)
                #img_vol = read_mri_volume(fid[fid_info]['filepath'])
                #print(img_vol.shape)

                #fig, ax = plt.subplots()
                #ax.imshow(img_vol[idx[1]-40:idx[1]+40,
                #    idx[0]-40:idx[0]+40, idx[2]], cmap='gray')
                #plt.show()

                # Extract metadata features.
                if len(metadata_features) == 0:
                    metadata_features.extend(extract_metadata_features(
                        fid[fid_info]['filepath']))

            # Process sagittal T2-weighted images and
            # extract metadata features.
            elif 't2_tse_sag0' in fid_info:
                idx = fid[fid_info]['finding_idx']

                # Extract metadata features.
                if len(metadata_features) == 0:
                    metadata_features.extend(extract_metadata_features(
                        fid[fid_info]['filepath']))


            elif 'ep2d_diff_tra_DYNDIST_ADC0' in fid_info:
                idx = fid[fid_info]['finding_idx']

                # Extract metadata features.
                if len(metadata_features) == 0:
                    metadata_features.extend(extract_metadata_features(
                        fid[fid_info]['filepath']))


            elif 'ep2d_diff_tra_DYNDISTCALC_BVAL0' in fid_info:
                idx = fid[fid_info]['finding_idx']

                # Extract metadata features.
                if len(metadata_features) == 0:
                    metadata_features.extend(extract_metadata_features(
                        fid[fid_info]['filepath']))

        fid['features'] = metadata_features + dce_features
        num_features = len(fid['features'])
        #print(fid['patient_name'], fid['id'], fid['features'])


    ## Predict whether or not lesions are malignant.
    findings_list = list(findings.keys())

    X = np.zeros((len(findings), num_features))
    y = np.zeros(len(findings))

    for i, finding in enumerate(findings_list):
        X[i, :] = findings[finding]['features']
        y[i] = findings[finding]['score']


    # Normalize all feature vectors, split into training and test data.
    X = preprocessing.normalize(X, norm='l2')

    ## Feature selection.
    selector = SelectKBest(f_classif, k='all')
    X_transform = selector.fit_transform(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_transform, y,
        test_size=0.2)


    # Optimize hyperparameters using 10-fold cross-validation.
    C_search = [1, 5, 10, 50, 100, 500, 1000]
    gamma_search = [1e-2, 1e-3, 1e-4, 1e-5]
    degree_search = [2, 3, 4, 5]
    param_grid = [{'kernel': ['linear'], 'C': C_search},
                  {'kernel': ['poly'], 'gamma': gamma_search,
                   'degree': degree_search, 'C': C_search},
                  {'kernel': ['rbf'], 'gamma': gamma_search,
                   'C': C_search}]

    #print('Estimating best hyperparameters...')
    #best_params = optimize_svc_params(X_train, y_train, param_grid)
    #svc = svm.SVC(**best_params, probability=True)

    svc = svm.SVC(kernel='rbf', C=500, gamma=0.0001, class_weight='balanced',
        probability=True)

    svc.fit(X_train, y_train)
    y_predicted = svc.predict(X_test)

    true_positives = np.sum((y_predicted == y_test) & (y_test == 1))
    true_negatives = np.sum((y_predicted == y_test) & (y_test == 0))
    positives = np.sum(y_test == 1)
    negatives = np.sum(y_test == 0)
    print('Accuracy = {:.2f}%'.format(100*np.mean(y_predicted == y_test)))
    print('Sensitivity = {:.2f}%'.format(100*true_positives/positives))
    print('Specificity = {:.2f}%'.format(100*true_negatives/negatives))



if __name__ == '__main__':
    main()
