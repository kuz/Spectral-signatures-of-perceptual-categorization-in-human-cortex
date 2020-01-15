import os
import numpy as np
import scipy.io as sio
from scipy.stats import mannwhitneyu
import pdb

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
DATADIR = '../../Data/Intracranial/Processed'
CLUSTDIR = '../../Outcome/Clustering'


# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# separate plot for each category
for cid, category in enumerate(categories):

    print '--- %s ---' % category

    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    successful_areas = np.load('%s/%s' % (INDIR, 'FT_successful_areas_ctg%d.npy' % cid))
    cluster_labels = np.load('%s/%d-%s/successful_probes_to_cluster_labels.npy' % (CLUSTDIR, cid, categories[cid]))
    important_activity_patterns = np.load('%s/%d-%s/important_activity_patterns.npy' % (CLUSTDIR, cid, categories[cid]))

    # drop 0,0,0 electrode if it creeps in
    drop_idx = np.unique(np.where(successful_mnis == [0, 0, 0])[0])
    importance = np.delete(importance, drop_idx, 0)
    successful_probes = np.delete(successful_probes, drop_idx, 0)
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)
    successful_areas = np.delete(successful_areas, drop_idx, 0)
    cluster_labels = np.delete(cluster_labels, drop_idx, 0)
    important_activity_patterns = np.delete(important_activity_patterns, drop_idx, 0)

    # Across all categories, the earliest component that often appeared in clusters was
    # the brief power increase in the low-frequency interval (4-25 Hz)
    #roi = np.ravel(important_activity_patterns[:, :21, 16:24])
    #print np.exp(np.mean(roi[roi > 0.0]))

    # which for one group of probes can be associated to an almost instantaneous broadband
    # gamma power increase (\ref{fig:importances-clusters-mnis}b, cluster 3
    if category == 'visage':
        roi = important_activity_patterns[cluster_labels == 3]
        roi = np.ravel(important_activity_patterns[:, 56:, 19:29])

