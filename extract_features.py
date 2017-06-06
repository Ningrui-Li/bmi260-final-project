def extract_metadata_features(dcm):
    '''
    Given a PyDICOM dataset object, extract random metadata features from the
    patient, like his age.
    '''
    # Add patient's age, size, and weight.
    features = [int(dcm.PatientAge[:-1]), dcm.PatientSize, dcm.PatientWeight]

    return features
