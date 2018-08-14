import os
import numpy as np
from functions_plotting import triptych_importance, triptych_fitfdiff
from functions_helpers import safemkdir
from matplotlib import pylab as plt
import scipy.io as sio
from scipy.stats import mannwhitneyu
import pdb

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures/polymono'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# global mono- and polypredictive probes
all_importance_mono = None
all_importance_poly = None
all_mnis_mono = np.empty((0, 3))
all_mnis_poly = np.empty((0, 3))

# separate plot for each category
for cid, category in enumerate(categories):

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

    # common importance
    #importance_threshold_mono = np.mean(importance[monoprobes, :, :]) + np.std(importance[monoprobes, :, :])
    #importance_threshold_poly = np.mean(importance[polyprobes, :, :]) + np.std(importance[polyprobes, :, :])
    importance_threshold_mono = 4 * np.std(importance[monoprobes, :, :15])
    importance_threshold_poly = 4 * np.std(importance[polyprobes, :, :15])
    most_importantce_mono = np.array(np.mean(importance[monoprobes, :, :], axis = 0) > importance_threshold_mono, dtype=np.int8)
    most_importantce_poly = np.array(np.mean(importance[polyprobes, :, :], axis = 0) > importance_threshold_poly, dtype=np.int8)
    common_importantce = np.array(most_importantce_mono + most_importantce_poly >= 2, dtype=np.int8)

    # sigma difference
    sigma = 4.0
    mean_importance_mono = np.mean(importance[monoprobes, :, :], axis=0)
    mean_importance_poly = np.mean(importance[polyprobes, :, :], axis=0)
    mean_importance_mono /= np.max(mean_importance_mono)
    mean_importance_poly /= np.max(mean_importance_poly)
    diff = mean_importance_mono - mean_importance_poly
    absdiff = np.abs(diff)
    significant_diff = absdiff > np.std(absdiff) * sigma

    diffmap = common_importantce
    diffmap[np.where(diff * significant_diff > 0)] =  2
    diffmap[np.where(diff * significant_diff < 0)] = -1
    both_mnis = np.vstack((successful_mnis[monoprobes], successful_mnis[polyprobes]))
    triptych_fitfdiff(diffmap, both_mnis, ['red'] * len(successful_mnis[monoprobes]) + ['blue'] * len(successful_mnis[polyprobes]),
                      {'red': 0.5, 'blue': 0.5}, ["%s/poly_vs_mono_%d_%s.png" % (OUTDIR, cid, category)],
                      title="%d sigma difference: poly/mono FITF maps of %s" % (sigma, category), lines=True)


# mono and polu over all categories
triptych_importance(np.mean(all_importance_mono, axis=0), all_mnis_mono, ['red'] * len(all_mnis_mono),
                    {'red': 0.3}, ["%s/monopredictive_all_categories.png" % OUTDIR],
                    title="Monopredictive probes (%d) across all categories" % all_mnis_mono.shape[0], lines=True)
triptych_importance(np.mean(all_importance_poly, axis=0), all_mnis_poly, ['blue'] * len(all_mnis_poly),
                    {'blue': 0.3}, ["%s/polypredictive_all_categories.png" % OUTDIR],
                    title="Polypredictive probes (%d) across all categories" % all_mnis_poly.shape[0], lines=True)

# not sigma-different importance
importance_threshold_mono = 4 * np.std(all_importance_mono[:, :, :15])
importance_threshold_poly = 4 * np.std(all_importance_poly[:, :, :15])
most_importantce_mono = np.array(np.mean(all_importance_mono, axis = 0) > importance_threshold_mono, dtype=np.int8)
most_importantce_poly = np.array(np.mean(all_importance_poly, axis = 0) > importance_threshold_poly, dtype=np.int8)
common_importantce = np.array(most_importantce_mono + most_importantce_poly >= 1, dtype=np.int8)

# sigma difference
sigma = 4.0
mean_importance_mono = np.mean(all_importance_mono, axis=0)
mean_importance_poly = np.mean(all_importance_poly, axis=0)
mean_importance_mono /= np.max(mean_importance_mono)
mean_importance_poly /= np.max(mean_importance_poly)
diff = mean_importance_mono - mean_importance_poly
absdiff = np.abs(diff)
significant_diff = absdiff > np.std(absdiff) * sigma

diffmap = common_importantce
diffmap[np.where(diff * significant_diff > 0)] =  2
diffmap[np.where(diff * significant_diff < 0)] = -1
both_mnis = np.vstack((all_mnis_mono, all_mnis_poly))
triptych_fitfdiff(diffmap, both_mnis, ['red'] * len(all_mnis_mono) + ['blue'] * len(all_mnis_poly),
                  {'red': 0.3, 'blue': 0.3}, ["%s/poly_vs_mono_all_categories.png" % OUTDIR],
                  title="%d sigma difference between poly and mono FITF maps" % sigma, lines=True) 



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
