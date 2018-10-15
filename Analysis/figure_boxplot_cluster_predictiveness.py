"""

To setup Freesurfer run
$ export FREESURFER_HOME=/Applications/freesurfer
$ source $FREESURFER_HOME/SetUpFreeSurfer.sh

"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from scipy.stats import mannwhitneyu
import pdb

from functions_plotting import panel_boxplots
from functions_helpers import safemkdir

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# separate subplot for each category
plot_data = {}
for cid, category in enumerate(categories):
    plot_data[cid] = {'f1': [], 'sig': np.zeros(6)}

    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    cluster_labels = np.load('%s/%d-%s/successful_probes_to_cluster_labels.npy' % (CLUSTDIR, cid, categories[cid]))
    important_activity_patterns = np.load('%s/%d-%s/important_activity_patterns.npy' % (CLUSTDIR, cid, categories[cid]))

    # drop 0,0,0 electrode if it creeps in
    drop_idx = np.unique(np.where(successful_mnis == [0, 0, 0])[0])
    successful_probes = np.delete(successful_probes, drop_idx, 0)
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)
    cluster_labels = np.delete(cluster_labels, drop_idx, 0)
    important_activity_patterns = np.delete(important_activity_patterns, drop_idx, 0)

    # four most important clusters' sample means, we manually specify for every category which cluster ID's are the most important ones
    important_clusters = {
        0: [1, 4, 5, 6],
        1: [1, 2, 3, 7],
        2: [1, 3, 6, 10],
        3: [1, 2, 3, 4],
        4: [1, 2, 5, 9],
        5: [1, 2, 8, 9],
        6: [1, 5, 9, 10],
        7: [1, 2, 3, 5]
    }

    # Category mean, 4 clusters, locations
    colors = ['whitesmoke', 'green', 'blue', 'red', 'black']
    cluster_means = np.zeros((4, important_activity_patterns.shape[1], important_activity_patterns.shape[2]))
    cluster_predictive_score = np.zeros((4,))
    cluster_probe_f1_scores = {0: [], 1: [], 2: [], 3: []}
    for i in range(4):

        # compute cluster's mean activity
        cluster_means[i] = np.mean(important_activity_patterns[cluster_labels == important_clusters[cid][i]], axis=0)

        # compute cluster's average predictive score (over probes in the cluster)
        sum_scores = 0
        ind_scores = []
        for succ_pid, succ_sid in successful_probes[cluster_labels == important_clusters[cid][i]]:
            succ_pid -= 1
            sum_scores += scores_spc[succ_sid][succ_pid][cid]
            cluster_probe_f1_scores[i].append(scores_spc[succ_sid][succ_pid][cid])
        if successful_probes[cluster_labels == important_clusters[cid][i]].shape[0] > 0:
            cluster_predictive_score[i] = sum_scores / successful_probes[cluster_labels == important_clusters[cid][i]].shape[0]
        else:
            cluster_predictive_score[i] = 0.0

        plot_data[cid]['f1'].append(cluster_probe_f1_scores[i])

    # difference significance
    '''
    diff_pvalue_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            diff_pvalue_matrix[i, j] = mannwhitneyu(cluster_probe_f1_scores[i], cluster_probe_f1_scores[j])[1] * (4*16)
    
    np.set_printoptions(precision=6, suppress=True)
    print diff_pvalue_matrix
    print ""
    '''
    plot_data[cid]['sig'][0] = mannwhitneyu(cluster_probe_f1_scores[0], cluster_probe_f1_scores[1])[1]
    plot_data[cid]['sig'][1] = mannwhitneyu(cluster_probe_f1_scores[1], cluster_probe_f1_scores[2])[1]
    plot_data[cid]['sig'][2] = mannwhitneyu(cluster_probe_f1_scores[2], cluster_probe_f1_scores[3])[1]
    plot_data[cid]['sig'][3] = mannwhitneyu(cluster_probe_f1_scores[0], cluster_probe_f1_scores[2])[1]
    plot_data[cid]['sig'][4] = mannwhitneyu(cluster_probe_f1_scores[1], cluster_probe_f1_scores[3])[1]
    plot_data[cid]['sig'][5] = mannwhitneyu(cluster_probe_f1_scores[0], cluster_probe_f1_scores[3])[1]


fig = plt.figure(figsize=(10, 5), dpi=200);
for cid in range(8):
    ax = plt.subplot2grid((2, 4), (int(cid / 4), cid % 4), colspan=1, rowspan=1)
    panel_boxplots(plot_data[cid]['f1'], plot_data[cid]['sig'], categories[cid], cid in [4, 5, 6, 7], cid in [0, 4])
    
plt.savefig('%s/boxplots/predictiveness_clusters.png' % (OUTDIR, ), bbox_inches='tight');
plt.clf();
plt.close(fig);




    

