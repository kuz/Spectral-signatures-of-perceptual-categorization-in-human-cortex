import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import pdb
from functions_plotting import panel_time_profile

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures/time_histogram'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))

# global counts
all_counts = np.zeros(48)

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
    
    # count 4-sigma important probes in each time bin
    importance_threshold = np.mean(importance) + 4 * np.std(importance)
    ctg_counts = np.zeros(48)
    for pid in range(successful_probes.shape[0]):
        most_importantce = np.array(importance[pid] > importance_threshold, dtype=np.int8)
        ctg_counts += np.array(np.sum(most_importantce, axis=0) > 0, dtype=np.int8)
        all_counts += np.array(np.sum(most_importantce, axis=0) > 0, dtype=np.int8)

    fig = plt.figure(figsize=(10, 8), dpi=200);
    panel_time_profile(ctg_counts, "Number of probes in time that are predictive of %s" % category, lines=True)
    plt.savefig('%s/time_count_histogram_%d_%s.png' % (OUTDIR, cid, category), bbox_inches='tight');
    plt.clf();
    plt.close(fig);

# for all categories
fig = plt.figure(figsize=(10, 8), dpi=200);
panel_time_profile(all_counts, "Number of predictive probes in time over all categoires", ylabel="Number of probes", lines=True)
plt.savefig('%s/time_count_histogram_all.png' % OUTDIR, bbox_inches='tight');
plt.clf();
plt.close(fig);
