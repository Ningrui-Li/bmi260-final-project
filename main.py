def read_patient_labels(findings_filename, images_filename):
    '''
    INPUT:
    findings_filename - name of CSV file with labels for each finding.
    images_filename - name of CSV file with info for each image acquisition.

    OUTPUTS:
    patients - dictionary that maps patient names to another dictionary
               containing information about each patient (like their name,
               label, etc.) Patients are labeled as 1 if the tumor is malignant
               and 0 otherwise.
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

            

    return patients


def main():
    findings_filename = 'ProstateX-2-Findings-Train.csv'
    images_filename = 'ProstateX-2-Images-Train.csv'

    patients = read_patient_labels(findings_filename, images_filename)
    print(patients)

if __name__ == '__main__':
    main()
