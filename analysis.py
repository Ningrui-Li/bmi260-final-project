import matplotlib.pyplot as plt

from helper import *
from extract_features import *

"""
def analyze_histogram_features(findings):
    '''
    Analyzes performance of histogram-based features for predicting lesion malignancy. Makes plots too!
    '''
"""

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
