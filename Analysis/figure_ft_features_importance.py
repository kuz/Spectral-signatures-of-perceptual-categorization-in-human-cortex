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
from surfer import Brain
import pdb

from functions_plotting import panel_importance
from functions_helpers import safemkdir

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

# surfer parameters
subject_id = "fsaverage"
#subjects_dir = os.environ["SUBJECTS_DIR"]

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# plot definition
def cluster_mean(activity, title, cluster_color):
    fig = plt.figure(figsize=(8, 6), dpi=300);
    plt.imshow(d, interpolation='none', origin='lower', cmap=cm.bwr, aspect='auto', vmin=-3.0, vmax=3.0);
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=12, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=12);
    plt.ylabel('Frequency (Hz)', size=16);
    plt.xlabel('Time (ms)', size=16);
    plt.title(title, size=16);
    plt.savefig(fname, bbox_inches='tight');
    plt.clf();
    plt.close(fig);

def quadriptych(importances, foci, foci_colors, cluster_means, cluster_predictive_score, title, filenames, lines=True):
    
    fig = plt.figure(figsize=(40, 8), dpi=300);
    vlim = np.max([np.abs(np.min(cluster_means)), np.abs(np.max(cluster_means))]) * 1.2

    # overall importances
    ax1 = plt.subplot2grid((2, 8), (0, 0), colspan=2, rowspan=2)
    panel_importance(importances, title, lines)

    # 4 most prominents clusters of activity under the important regions
    ax2 = plt.subplot2grid((2, 8), (0, 2))
    plt.imshow(cluster_means[0], interpolation='none', origin='lower', cmap=cm.bwr, aspect=0.3, vmin=-vlim, vmax=vlim);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=8, rotation=90);
    plt.yticks(np.arange(0, 146, 10), np.arange(4, 150, 10), size=9);
    plt.ylabel('Frequency (Hz)', size=16);
    plt.title('Activity of GREEN electrodes', size=14, color='green');
    ax2.text(1, 129, '1', fontsize=20)
    ax2.text(35, 132, '%.4f' % cluster_predictive_score[0], fontsize=15)

    ax1 = plt.subplot2grid((2, 8), (0, 3))
    plt.imshow(cluster_means[1], interpolation='none', origin='lower', cmap=cm.bwr, aspect=0.3, vmin=-vlim, vmax=vlim);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=8, rotation=90);
    plt.yticks(np.arange(0, 146, 10), np.arange(4, 150, 10), size=9);
    plt.title('Activity of BLUE electrodes', size=14, color='blue');
    ax1.text(1, 129, '2', fontsize=20)
    ax1.text(35, 132, '%.4f' % cluster_predictive_score[1], fontsize=15)

    ax1 = plt.subplot2grid((2, 8), (1, 2))
    plt.imshow(cluster_means[2], interpolation='none', origin='lower', cmap=cm.bwr, aspect=0.3, vmin=-vlim, vmax=vlim);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=8, rotation=90);
    plt.yticks(np.arange(0, 146, 10), np.arange(4, 150, 10), size=9);
    plt.ylabel('Frequency (Hz)', size=16);
    plt.xlabel('Time (ms)', size=16);
    plt.title('Activity of RED electrodes', size=14, color='red');
    ax1.text(1, 129, '3', fontsize=20)
    ax1.text(35, 132, '%.4f' % cluster_predictive_score[2], fontsize=15)

    ax1 = plt.subplot2grid((2, 8), (1, 3))
    plt.imshow(cluster_means[3], interpolation='none', origin='lower', cmap=cm.bwr, aspect=0.3, vmin=-vlim, vmax=vlim);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48, 2), np.asarray((np.arange(0, 769, 32) - 256) / 512.0 * 1000, dtype='int'), size=8, rotation=90);
    plt.yticks(np.arange(0, 146, 10), np.arange(4, 150, 10), size=9);
    plt.xlabel('Time (ms)', size=16);
    plt.title('Activity of BLACK electrodes', size=14, color='black');
    ax1.text(1, 129, '4', fontsize=20)
    ax1.text(35, 132, '%.4f' % cluster_predictive_score[3], fontsize=15)

    # 3D mesh
    ax1 = plt.subplot2grid((2, 8), (0, 4), colspan=2, rowspan=2)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.6, color=color);
    brain.show_view('m');
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlabel('MNI Y', size=16);
    plt.xticks(np.arange(0, 800, 20), np.asarray(np.arange(-75, 106, 4.5), dtype='int'), size=10, rotation=90);
    plt.ylabel('MNI Z', size=16);
    plt.yticks(np.arange(0, 800, 20), np.asarray(np.arange(96, -87, -4.372), dtype='int'), size=10);
    plt.xlim(0, 800);
    plt.ylim(800, 0);
    ax1.text(10, 39, '1', fontsize=20)

    # 3D mesh
    ax1 = plt.subplot2grid((2, 8), (0, 6), colspan=2, rowspan=2)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.6, color=color);
    brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlabel('MNI Y', size=16);
    plt.xticks(np.arange(0, 800, 20), np.asarray(np.arange(-75, 106, 4.5), dtype='int'), size=10, rotation=90);
    plt.ylabel('MNI X', size=16);
    plt.yticks(np.arange(0, 800, 20), np.asarray(np.arange(-94.5, 93.5, 4.7), dtype='int'), size=10);
    plt.xlim(0, 800);
    plt.ylim(800, 0);
    ax1.text(10, 39, '2', fontsize=20)

    # store the figure
    for filename in filenames:
        plt.savefig(filename, bbox_inches='tight');
    plt.clf();
    plt.close(fig);


# separate plot for each category
for cid, category in enumerate(categories):

    print 'Drawing "%s" ...' % category
    subdir = 'FT_importances_%d_%s' % (cid, category)
    safemkdir('%s/%s' % (OUTDIR, subdir))

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

    '''# drop all that are not showing broadband gamma decrease
    bbgamma_decrease = [(3, 14), (3, 102), (3, 104), (15, 117), (15, 119), (15, 154), (22, 91), (24, 49), (24, 138), (33, 116), (42, 107),
                        (57, 148), (75, 16), (80, 90), (80, 91), (1, 107), (6, 74), (8, 105), (9, 12), (9, 29), (9, 37), (15, 118), (15, 155),
                        (22, 81), (22, 89), (24, 132), (28, 72), (30, 33), (33, 117), (33, 121), (33, 122), (38, 92), (40, 122), (49, 69),
                        (54, 7), (57, 115), (58, 99), (58, 100), (60, 81), (60, 83), (66, 114), (73, 70), (75, 15), (75, 20), (75, 83),
                        (75, 84), (76, 29), (79, 16), (79, 91), (79, 92), (79, 93), (80, 89), (81, 52), (81, 53), (81, 75), (81, 76),
                        (82, 62), (87, 65), (87, 100), (89, 30), (90, 82), (90, 83), (90, 93), (91, 44), (94, 54), (94, 55), (94, 86)]
    drop_idx = []
    for i, (pid, sid) in enumerate(successful_probes):
        if (sid, pid) not in bbgamma_decrease: # order of pid and sid is reversed here on purpose
            drop_idx.append(i)
    importance = np.delete(importance, drop_idx, 0)
    successful_probes = np.delete(successful_probes, drop_idx, 0)
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)
    successful_areas = np.delete(successful_areas, drop_idx, 0)
    cluster_labels = np.delete(cluster_labels, drop_idx, 0)
    important_activity_patterns = np.delete(important_activity_patterns, drop_idx, 0)'''

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
    cluster_color_ids = np.array([0 for x in range(len(cluster_labels))])
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

        # assign color
        cluster_color_ids[cluster_labels == important_clusters[cid][i]] = i + 1

    # power amplitudes
    for i in range(4):
        print "Cluster %d: %.4f" % (i + 1, np.percentile(cluster_means[i, 46:, 20:35], 75))

    '''# difference significance
    diff_pvalue_matrix = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            diff_pvalue_matrix[i, j] = mannwhitneyu(cluster_probe_f1_scores[i], cluster_probe_f1_scores[j])[1] * (8*16)
    
    np.set_printoptions(precision=6, suppress=True)
    print diff_pvalue_matrix
    print ""'''

    # generate figures
    foci_colors = np.array([colors[i] for i in cluster_color_ids])
    #importance -= np.percentile(importance[:, :15, :], 90) # normalize by subtracting almost MAX of aseline
    #importance[importance < 0.0] = 0.0 
    #importance /= np.sum(importance)
    quadriptych(np.mean(importance, 0), successful_mnis, foci_colors, cluster_means, cluster_predictive_score, 
                'Importance of spectrotemporal features for "%s"' % categories[cid],
                ['%s/FT_importances_%d_%s_MEAN.png' % (OUTDIR, cid, category)])

    # importance in time
    #most_important_moment = np.argmax(np.sum(importance[:, :, :], axis=1), axis=1)
    #most_important_moment = np.argsort(np.sum((importance[:, :, 17:48] * temporal_weights), axis=(1, 2)))
    #temporal_weights = np.tile(range(17,48), len(successful_areas)*146).reshape(len(successful_areas), 146, len(range(17,48))) / 47.0

    '''# Plot each probe's importances
    for i in range(importance.shape[0]):

        (pid, sid) = successful_probes[i]
        pid = pid - 1
        area = successful_areas[i]
        mni = successful_mnis[i]
        
        # each BA into own directory
        try:
            os.mkdir('%s/%s/BA%d' % (OUTDIR, subdir, area))
        except:
            pass

        # each Subject into own directory
        try:
            os.mkdir('%s/%s/Subject%d' % (OUTDIR, subdir, sid))
        except:
            pass

        triptych(importance[i, :, :], mni,
                 'Importance of spectrotemporal features for "%s"\nsID: %d   pID: %d   BA: %d' % (categories[cid], sid, pid, area),
                 ['%s/%s/FT_importances_%d_%s_s%d-p%d.png' % (OUTDIR, subdir, cid, category, sid, pid),
                  '%s/%s/BA%d/FT_importances_%d_%s_s%d-p%d.png' % (OUTDIR, subdir, area, cid, category, sid, pid),
                  '%s/%s/Subject%d/FT_importances_%d_%s_s%d-p%d.png' % (OUTDIR, subdir, sid, cid, category, sid, pid)])'''

    '''# 
    for ba in np.unique(successful_areas):

        #most_important_moment = np.argmax(np.sum(np.mean(importance[successful_areas == ba, :, :], axis=0), axis=0))

        baprobes = np.mean(importance[successful_areas == ba, :, :], axis=0)
        maxfreqs = baprobes[:, 0].argsort()[::-1][:10]
        most_important_moment = np.argmax(np.sum(np.mean(importance[successful_areas == ba, :, :], axis=0)[maxfreqs, :], axis=0))

        fig = plt.figure(figsize=(16, 6), dpi=300);
        plt.subplot(1, 2, 1);
        plt.imshow(np.mean(importance[successful_areas == ba, :, :], 0), interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
        plt.colorbar();
        plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
        plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
        plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
        plt.ylabel('Frequency (Hz)', size=10);
        plt.xlabel('Time', size=10);
        plt.title('Importance of spectrotemporal features for "%s"\nmean over %d probes in BA%d' % (categories[cid], np.sum(successful_areas == ba), ba), size=11);

        # 3D mesh
        plt.subplot(1, 2, 2);
        brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
        brain.show_view(view=dict(azimuth=152.88, elevation=65.94), roll=101.58);
        brain.add_foci(successful_mnis[successful_areas == ba], hemi='rh', scale_factor=0.5, color='blue');
        pic = brain.screenshot()
        plt.imshow(pic);

        plt.savefig('%s/%s/Time/FT_importances_%d_%s_time%d_MEAN_BA%d.png' % (OUTDIR, subdir, cid, category, most_important_moment, ba), bbox_inches='tight');
        plt.clf();
        plt.close(fig);'''

    '''# Mean over each BA
    for ba in np.unique(successful_areas):
        fig = plt.figure(figsize=(10, 8), dpi=300);
        title = 'Importance of spectrotemporal features for "%s"\nmean over %d probes in BA%d' % (categories[cid], np.sum(successful_areas == ba), ba)
        panel_importance(np.mean(importance[successful_areas == ba, :, :], 0), title, True)
        plt.savefig('%s/FT_importances_%d_%s_BA%d.png' % (OUTDIR, cid, category, ba), bbox_inches='tight');
        plt.clf();
        plt.close(fig);'''
