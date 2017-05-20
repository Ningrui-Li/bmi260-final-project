import csv
from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import dicom as dcm


def read_patient_labels(findings_filename, images_filename, mri_dir, dce_dir):
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
    patients = dict()

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

            if finding['patient_name'] not in patients:
                patients[finding['patient_name']] = dict()
            patients[finding['patient_name']][finding['id']] = finding

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


            # Get filepath to images acquired for a given pulse sequence.
            if seq_name not in patients[patient_name][fid]:
                patients[patient_name][fid][seq_name] = dict()

            patients[patient_name][fid][seq_name]['filepath'] = join(
                patient_mri_dir, seq_id)
            patients[patient_name][fid][seq_name]['world_matrix'] = line[5]
            patients[patient_name][fid][seq_name]['finding_idx'] = idx
            patients[patient_name][fid][seq_name]['vox_spacing'] = spacing

            # Get filepath to Ktrans map.
            patients[patient_name][fid]['dce'] = dict()
            patients[patient_name][fid]['dce']['filepath'] = join(
                get_dce_dir(dce_dir, patient_name))

    return patients


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


def main():
    mri_dir = 'DOI'
    dce_dir = 'KtransTrain'
    findings_filename = 'ProstateX-2-Findings-Train.csv'
    images_filename = 'ProstateX-2-Images-Train.csv'

    #print(listdir(get_mri_dir(mri_dir, 'ProstateX-0014')))
    #print(listdir(get_dce_dir(dce_dir, 'ProstateX-0014')))
    #return

    patients = read_patient_labels(findings_filename, images_filename,
        mri_dir, dce_dir)


    for patient in patients:
        for fid in patients[patient]:
            #print(patient, patients[patient][fid])
            print(patients[patient][fid]['dce'])
            return


if __name__ == '__main__':
    main()
