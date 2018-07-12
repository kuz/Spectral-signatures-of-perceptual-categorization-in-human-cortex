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

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

# surfer parameters
subject_id = "fsaverage"

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

def safemkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def triptych(importances, foci, title, filenames, lines=True):
    
    fig = plt.figure(figsize=(28, 8), dpi=200);

    # overall importances
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=1, rowspan=1)
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48, 1.5), np.asarray((np.arange(0, 769, 24) - 256) / 512.0 * 1000, dtype='int'), size=11, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=12);
    plt.ylabel('Frequency (Hz)', size=16);
    plt.xlabel('Time (ms)', size=16);
    plt.title(title, size=16);

    # 3D mesh
    ax1 = plt.subplot2grid((1, 3), (0, 1), colspan=1, rowspan=1)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    brain.add_foci(foci, hemi='rh', scale_factor=0.4, color="blue");
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
    ax1 = plt.subplot2grid((1, 3), (0, 2), colspan=1, rowspan=1)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    brain.add_foci(foci, hemi='rh', scale_factor=0.4, color="blue");
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

# global mono- and polypredictive probes
all_importance_mono = None
all_importance_poly = None
all_mnis_mono = np.empty((0, 3))
all_mnis_poly = np.empty((0, 3))

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

    
    # Mono and Poly predictive probes (of this catergory)
    monoprobes = []
    polyprobes = []
    for i, (pid, sid) in enumerate(successful_probes):
        pid = pid - 1
        if len(scores_spc[sid][pid]) == 1:
            monoprobes.append(i)
        else:
            polyprobes.append(i)

    if all_importance_mono is None:
        all_importance_mono = np.empty((0, importance.shape[1], importance.shape[2]))
        all_importance_poly = np.empty((0, importance.shape[1], importance.shape[2]))

    all_importance_mono = np.vstack((all_importance_mono, importance[monoprobes, :, :]))
    all_importance_poly = np.vstack((all_importance_poly, importance[polyprobes, :, :]))
    all_mnis_mono = np.vstack((all_mnis_mono, successful_mnis[monoprobes]))
    all_mnis_poly = np.vstack((all_mnis_poly, successful_mnis[polyprobes]))

triptych(np.mean(all_importance_mono, axis=0), all_mnis_mono, "Monopredictive probes across all categories",
                 ["%s/Monopredictive_all_categoires.png" % OUTDIR])
triptych(np.mean(all_importance_poly, axis=0), all_mnis_poly, "Polypredictive probes across all categories",
                 ["%s/Polypredictive_all_categoires.png" % OUTDIR])

print "Number of monopredictive: %d" % all_mnis_mono.shape[0]
print "Number of polypredictive: %d" % all_mnis_poly.shape[0]
pdb.set_trace()


if False:
    pass
    """
    triptych(np.mean(importance[monoprobes, :, :], 0), successful_mnis[monoprobes],
             'Importance of spectrotemporal features for "%s"\nmean over %d monopredictive probes' % (categories[cid], len(monoprobes)),
             ['%s/%s/FT_importances_%d_%s_MONO_MEAN.png' % (OUTDIR, subdir, cid, category)])
    print 'BAs: ', successful_areas[monoprobes]

    triptych(np.mean(importance[polyprobes, :, :], 0), successful_mnis[polyprobes],
             'Importance of spectrotemporal features for "%s"\nmean over %d polypredictive probes' % (categories[cid], len(polyprobes)),
             ['%s/%s/FT_importances_%d_%s_POLY_MEAN.png' % (OUTDIR, subdir, cid, category)])
    print 'BAs: ', successful_areas[polyprobes]
    """
    

    """
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
        plt.close(fig);
    """

    # Mean over each BA
    """
    for ba in np.unique(successful_areas):    
        triptych(np.mean(importance[successful_areas == ba, :, :], 0), successful_mnis[successful_areas == ba],
                 'Importance of spectrotemporal features for "%s"\nmean over %d probes in BA%d' % (categories[cid], np.sum(successful_areas == ba), ba),
                 ['%s/%s/FT_importances_%d_%s_MEAN_BA%d.png' % (OUTDIR, subdir, cid, category, ba)])
    """
