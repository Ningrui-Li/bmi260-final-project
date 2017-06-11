import matplotlib.pyplot as plt

from helper import *
from extract_features import *


def analyze_histogram_features(findings):
    '''
    Analyzes performance of histogram-based features for predicting lesion malignancy. Makes plots too!
    '''
    # Create list of histogram feature values for each score group.
    roi_max_dce = [[] for i in range(5)]
    roi_mean_dce = [[] for i in range(5)]
    roi_median_dce = [[] for i in range(5)]
    roi_std_dce = [[] for i in range(5)]

    img_max_dce = [[] for i in range(5)]
    img_mean_dce = [[] for i in range(5)]
    img_median_dce = [[] for i in range(5)]
    img_std_dce = [[] for i in range(5)]

    ktrans_finding_idx = [[] for i in range(5)]


    for _, fid in findings.items():
        dce_features = extract_dce_features(fid['dce']['filepath'],
            fid['pos'])
        roi_max_dce[fid['score']-1].append(dce_features[0])
        roi_mean_dce[fid['score']-1].append(dce_features[1])
        roi_median_dce[fid['score']-1].append(dce_features[2])
        roi_std_dce[fid['score']-1].append(dce_features[3])

        img_max_dce[fid['score']-1].append(dce_features[4])
        img_mean_dce[fid['score']-1].append(dce_features[5])
        img_median_dce[fid['score']-1].append(dce_features[6])
        img_std_dce[fid['score']-1].append(dce_features[7])

        ktrans_finding_idx[fid['score']-1].append(dce_features[8])


    ## Plots for ROI histogram stats.
    # Create boxplot for max KTrans value in ROI.
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(2, 2, 1)
    ax.boxplot(roi_max_dce)
    ax.set_title('Max $K_{trans}$ in ROI for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Max $K_{trans}$ in ROI')

    # Create boxplot for mean KTrans value in ROI.
    ax = fig.add_subplot(2, 2, 2)
    ax.boxplot(roi_mean_dce)
    ax.set_title('Mean $K_{trans}$ in ROI for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Mean $K_{trans}$ in ROI')

    # Create boxplot for median KTrans value in ROI.
    ax = fig.add_subplot(2, 2, 3)
    ax.boxplot(roi_median_dce)
    ax.set_title('Median $K_{trans}$ in ROI' + '\n' + \
        'for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Median $K_{trans}$ in ROI')

     # Create boxplot for standard deviation of KTrans values in ROI.
    ax = fig.add_subplot(2, 2, 4)
    ax.boxplot(roi_std_dce)
    ax.set_title('Standard Dev. $K_{trans}$ in ROI' + '\n' + \
        'for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Standard Dev. of $K_{trans}$ in ROI')

    plt.tight_layout()

    fig.savefig('plots/multiclass_ktrans_histogram_roi.png')


    ## Plots for histogram stats over entire image volume.
    # Create boxplot for max KTrans value.
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(2, 2, 1)
    ax.boxplot(img_max_dce)
    ax.set_title('Global Max $K_{trans}$ for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Global Max $K_{trans}$')

    # Create boxplot for mean KTrans value.
    ax = fig.add_subplot(2, 2, 2)
    ax.boxplot(img_mean_dce)
    ax.set_title('Global Mean $K_{trans}$ for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Global Mean $K_{trans}$')

    # Create boxplot for median KTrans value.
    ax = fig.add_subplot(2, 2, 3)
    ax.boxplot(img_median_dce)
    ax.set_title('Global Median $K_{trans}$' + '\n' + \
        'for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Global Median $K_{trans}$')

     # Create boxplot for standard deviation of KTrans values.
    ax = fig.add_subplot(2, 2, 4)
    ax.boxplot(img_std_dce)
    ax.set_title('Global Standard Dev. $K_{trans}$' + '\n' + \
        'for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'Global Standard Dev. of $K_{trans}$')

    plt.tight_layout()

    fig.savefig('plots/multiclass_ktrans_histogram_img.png')


    ## Plot for Ktrans value at the giving target location.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(ktrans_finding_idx)
    ax.set_title(r'$K_{trans}$ at Finding Location' + '\n' + \
        'for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel(r'$K_{trans}$ at Finding Location')

    fig.savefig('plots/multiclass_dce_ktrans_finding_idx.png')



    plt.show()
    plt.close('all')

    return


def analyze_metadata_features(findings):
    '''
    Analyzes performance of metadata features for predicting lesion malignancy.
    Makes plots too!
    '''
    # Create list of ages and weights for each score group.
    ages = [[] for i in range(5)]
    weights = [[] for i in range(5)]

    for _, fid in findings.items():
        metadata_features = extract_metadata_features(fid)
        ages[fid['score']-1].append(metadata_features[0])
        weights[fid['score']-1].append(metadata_features[1])

    # Create boxplot for ages.
    fig, ax = plt.subplots()
    ax.boxplot(ages)
    ax.set_title('Patient Ages for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel('Age (years)')
    fig.savefig('plots/multiclass_age.png')


    # Create boxplot for weights.
    fig, ax = plt.subplots()
    ax.boxplot(weights)
    ax.set_title('Patient Weights for Different Gleason Grade Groups')
    ax.set_xlabel('Gleason Grade Group')
    ax.set_ylabel('Weight (kg)')
    fig.savefig('plots/multiclass_weight.png')


    # Create 2D plot.
    fig, ax = plt.subplots()
    for i in range(5):
        ax.plot(ages[i], weights[i], '.', markersize=8,
            label='Gleason Group {}'.format(i+1))
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Weight (kg)')
    ax.legend()
    fig.savefig('plots/multiclass_age_vs_weight.png')

    plt.close('all')

    return
