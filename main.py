from sklearn import preprocessing, metrics, svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif

from extract_features import *
from helper import *

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


            # Process transverse T2-weighted images.
            elif 't2_tse_tra0' in fid_info:
                idx = fid[fid_info]['finding_idx']

                t2_features = extract_t2_features(fid[fid_info]['filepath'],
                    idx)

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
                adc_features = extract_adc_features(fid[fid_info]['filepath'],
                    idx)

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

        # Extract zone location as a feature.
        zone_features = extract_zone_features(fid['zone'])

        fid['features'] = metadata_features + zone_features
        fid['features'] += dce_features + t2_features + adc_features
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
