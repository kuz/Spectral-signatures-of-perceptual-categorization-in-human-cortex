import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import pdb
from functions_plotting import panel_curves

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures/time_curves'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# global mono- and polypredictive probes
all_importance = None

# separate plot for each category
for cid, category in enumerate(categories):

    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    
    # drop 0,0,0 electrode if it creeps in
    drop_idx = np.unique(np.where(successful_mnis == [0, 0, 0])[0])
    importance = np.delete(importance, drop_idx, 0)
    successful_probes = np.delete(successful_probes, drop_idx, 0)
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)
    
    if all_importance is None:
        all_importance = np.empty((0, importance.shape[1], importance.shape[2]))
    all_importance = np.vstack((all_importance, importance))

    # plot individual curves
    mean_importance_bbgamma = np.sum(np.mean(importance[:, 46:, :], axis=0), axis=0)
    mean_importance_lowfreq = np.sum(np.mean(importance[:, :46, :], axis=0), axis=0)
    mean_importance_bbgamma -= np.mean(mean_importance_bbgamma[:15])
    mean_importance_lowfreq -= np.mean(mean_importance_lowfreq[:15])
    
    fig = plt.figure(figsize=(10, 8), dpi=200);
    panel_curves([mean_importance_lowfreq, mean_importance_bbgamma],
                 ['-', '--'],
                 ['#348AA7', '#348AA7'],
                 ['4-50 Hz', '50-150Hz'],
                 'Time curves for "%s"' % category)
    plt.savefig('%s/time_curves_%d_%s.png' % (OUTDIR, cid, category), bbox_inches='tight');
    plt.clf();
    plt.close(fig);

# for all categories
mean_importance_bbgamma = np.sum(np.mean(all_importance[:, 46: :], axis=0), axis=0)
mean_importance_lowfreq = np.sum(np.mean(all_importance[:, :46 :], axis=0), axis=0)
mean_importance_bbgamma -= np.mean(mean_importance_bbgamma[:15])
mean_importance_lowfreq -= np.mean(mean_importance_lowfreq[:15])

fig = plt.figure(figsize=(10, 8), dpi=200);
panel_curves([mean_importance_lowfreq, mean_importance_bbgamma],
             ['-', '--'],
             ['#348AA7', '#348AA7'],
             ['4-50 Hz', '50-150Hz'],
             "Time curves for all categories")
plt.savefig('%s/time_curves_all.png' % OUTDIR, bbox_inches='tight');
plt.clf();
plt.close(fig);
