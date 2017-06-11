from os.path import isfile

from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np

from extract_features import *
from helper import *
from analysis import *


def compute_finding_statistics(findings):
    finding_grades = [0, 0, 0, 0, 0]
    for _, finding in findings.items():
        finding_grades[int(finding['score'])-1] += 1
    print(finding_grades)
    return


def compute_features(findings, multiclass):
    for _, fid in findings.items():
        print('Patient {}, finding {}'.format(fid['patient_name'], fid['id']))
        # Convert finding score to a binary value.
        fid['score'] = int(fid['score'])

        # Gleason grade groups 1 and 2 are considered "benign" if doing
        # binary classification.
        if not multiclass:
            if fid['score'] <= 2:
                fid['score'] = 0
            else:
                fid['score'] = 1

        # Extract patient metadata features.
        fid['metadata_features'] = extract_metadata_features(fid)

        # Convert finding position from string to coordinates.
        # Extract features from Ktrans images.
        fid['dce_features'] = extract_dce_features(
            fid['dce']['filepath'], fid['pos'])
        fid['dce_features'] = fid['dce_features'][:4] # keep only ROI features.

        # Extract features from transverse T2-weighted images.
        if 't2_tse_tra_Grappa30' in fid:
            t2_tra_name = 't2_tse_tra_Grappa30'
        else:
            t2_tra_name = 't2_tse_tra0'

        idx = fid[t2_tra_name]['finding_idx']
        fid['t2_features'] = extract_t2_features(
            fid[t2_tra_name]['filepath'], idx)

        # TODO: Extract features from sagittal T2-weighted images.
        # 't2_tse_sag0'

        # Extract features from ADC images.
        if 'ep2d_diff_tra_DYNDIST_ADC0' in fid:
            adc_name = 'ep2d_diff_tra_DYNDIST_ADC0'
        elif 'ep2d_diff_tra_DYNDIST_MIX_ADC0' in fid:
            adc_name = 'ep2d_diff_tra_DYNDIST_MIX_ADC0'
        elif 'ep2d_diff_tra2x2_Noise0_FS_DYNDIST_ADC0' in fid:
            adc_name = 'ep2d_diff_tra2x2_Noise0_FS_DYNDIST_ADC0'
        else:
            adc_name = 'diffusie_3Scan_4bval_fs_ADC0'

        idx = fid[adc_name]['finding_idx']
        fid['adc_features'] = extract_adc_features(
            fid[adc_name]['filepath'], idx)

        # TODO: Extract features from b-value images.
        #  'ep2d_diff_tra_DYNDISTCALC_BVAL0'

        # Extract zone location as a feature.
        fid['zone_features'] = extract_zone_features(fid['zone'])

        #print(fid['patient_name'], fid['id'], fid['features'])

    return findings

def main():
    # Set input parameters.
    multiclass = False
    mri_dir = 'DOI'
    dce_dir = 'KtransTrain'
    findings_filename = 'ProstateX-2-Findings-Train.csv'
    images_filename = 'ProstateX-2-Images-Train.csv'

    # Compute all features.
    # If features already computed, just load them in.
    if isfile('features.npy'):
        findings = np.load('features.npy').item()
    else:
        # Read in image information + labels.
        print('Reading labels...')
        findings = read_finding_labels(findings_filename, images_filename,
            mri_dir, dce_dir)
        print()

        print('Computing features...')
        findings = compute_features(findings, multiclass=False)

        np.save('features.npy', findings)

    compute_finding_statistics(findings)

    # Feature analysis.
    #print('Analyzing features...')
    #analyze_metadata_features(findings)
    #analyze_histogram_features(findings)
    #return

    ## Predict whether or not lesions are malignant.
    findings_list = list(findings.keys())
    num_features = len(findings[findings_list[0]]['metadata_features'])
    num_features += len(findings[findings_list[0]]['zone_features'])
    num_features += len(findings[findings_list[0]]['t2_features'])
    num_features += len(findings[findings_list[0]]['dce_features'])
    num_features += len(findings[findings_list[0]]['adc_features'])

    X = np.zeros((len(findings), num_features))
    y = np.zeros(len(findings))

    for i, finding in enumerate(findings_list):
        # Combine all features into a single vector.
        features = findings[finding]['metadata_features']
        features += findings[finding]['zone_features']
        features += findings[finding]['t2_features']
        features += findings[finding]['dce_features']
        features += findings[finding]['adc_features']

        X[i, :] = features
        y[i] = findings[finding]['score']

    print('Total number of features: ', num_features)

    # Normalize all feature vectors, split into training and test data.
    X = preprocessing.scale(X)

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

    print('Estimating best hyperparameters...')
    best_params = optimize_svc_params(X_train, y_train, param_grid)
    svc = svm.SVC(**best_params, class_weight='balanced', probability=True)

    #svc = svm.SVC(kernel='rbf', C=500, gamma=0.0001, class_weight='balanced',
    #    probability=True)

    svc.fit(X_train, y_train)
    y_predicted = svc.predict(X_test)

    true_positives = np.sum((y_predicted == y_test) & (y_test == 1))
    true_negatives = np.sum((y_predicted == y_test) & (y_test == 0))
    positives = np.sum(y_test == 1)
    negatives = np.sum(y_test == 0)
    print('Accuracy = {:.2f}%'.format(100*np.mean(y_predicted == y_test)))
    print('Sensitivity = {:.2f}%'.format(100*true_positives/positives))
    print('Specificity = {:.2f}%'.format(100*true_negatives/negatives))

    # Create ROC curve.
    y_prob_predicted = svc.predict_proba(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob_predicted[:,1],
        pos_label=1, drop_intermediate=False)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, 'k-')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title('ROC Curve for Classification with Semantic and Image'
        'Features\nSVC with RBF Kernel, C = 500, $\gamma$ = 0.0001')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig('roc_curve.png')

    auroc = metrics.roc_auc_score(y_test, y_prob_predicted[:,1])
    print('The area under the ROC curve is {:.3f}'.format(auroc))


if __name__ == '__main__':
    main()
