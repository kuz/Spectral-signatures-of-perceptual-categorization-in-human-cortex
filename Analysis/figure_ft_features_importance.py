import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from surfer import Brain

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

# surfer parameters
subject_id = "fsaverage"
subjects_dir = os.environ["SUBJECTS_DIR"]

# lists
#categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
categories = ['house']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

def safemkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

# plot definition
def cluster_mean(activity, title, cluster_color):
    fig = plt.figure(figsize=(8, 6), dpi=300);
    plt.imshow(d, interpolation='none', origin='lower', cmap=cm.bwr, aspect='auto', vmin=-3.0, vmax=3.0);
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
    plt.ylabel('Frequency (Hz)', size=10);
    plt.xlabel('Time (30 ms bin)', size=10);
    plt.title(title, size=11);
    plt.savefig(fname, bbox_inches='tight');
    plt.clf();
    plt.close(fig);


def quadriptych(importances, foci, foci_colors, title, filenames):

    fig = plt.figure(figsize=(32, 6), dpi=300);
    plt.subplot(1, 5, 1);
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, extent=(0, 100, 0, 100), aspect='equal');
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
    plt.ylabel('Frequency (Hz)', size=10);
    plt.xlabel('Time (30 ms bin)', size=10);
    plt.title(title, size=11);

    plt.subplot(2, 5, 2);
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, aspect=0.3);
    plt.subplot(2, 5, 3);
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, aspect=0.3);
    plt.subplot(2, 5, 7);
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, aspect=0.3);
    plt.subplot(2, 5, 8);
    plt.imshow(importances, interpolation='none', origin='lower', cmap=cm.Blues, aspect=0.3);

    # 3D mesh
    plt.subplot(1, 5, 4);
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.5, color=color);
    brain.show_view('m');
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlim(0, 800);
    plt.ylim(800, 0);

    # 3D mesh
    plt.subplot(1, 5, 5);
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
    for color in np.unique(foci_colors):
        brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.5, color=color);
    brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlim(0, 800);
    plt.ylim(800, 0);

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

    # four most important clusters' sample means
    # here we manually specify for every category which cluster ID's are the most important ones
    important_clusters_per_cid = {
        0: [1, 4, 5, 6],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
        6: [],
        7: []
    }
    important_clusters = important_clusters_per_cid[cid]

    # Category mean, 4 clusters, locations
    #colors = np.array(['blue' for x in range(successful_mnis.shape[0])])
    colors = np.array(['blue' for x in range(23)] + ['green' for x in range(23)])
    quadriptych(np.mean(importance, 0), successful_mnis, colors,
                'Importance of spectrotemporal features for "%s"' % categories[cid],
                ['%s/FT_importances_%d_%s_MEAN.png' % (OUTDIR, cid, category)])

    # importance in time
    #most_important_moment = np.argmax(np.sum(importance[:, :, :], axis=1), axis=1)
    #most_important_moment = np.argsort(np.sum((importance[:, :, 17:48] * temporal_weights), axis=(1, 2)))
    #temporal_weights = np.tile(range(17,48), len(successful_areas)*146).reshape(len(successful_areas), 146, len(range(17,48))) / 47.0

    # Plot each probe's importances
    """
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
                  '%s/%s/Subject%d/FT_importances_%d_%s_s%d-p%d.png' % (OUTDIR, subdir, sid, cid, category, sid, pid)])
    """

    # Mono and Poly predictive probes
    """
    monoprobes = []
    polyprobes = []
    for i, (pid, sid) in enumerate(successful_probes):
        pid = pid - 1
        if len(scores_spc[sid][pid]) == 1:
            monoprobes.append(i)
        else:
            polyprobes.append(i)

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
        plt.xlabel('Time (30 ms bin)', size=10);
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
