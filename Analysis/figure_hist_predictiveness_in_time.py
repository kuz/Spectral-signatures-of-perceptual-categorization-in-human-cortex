import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import pdb
from functions_plotting import panel_time_profile, duoptych

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
pid_showed_first_predictiveness_globally = {}
pid_showed_last_predictiveness_globally = {}

# probes at peaks
mnis_187 = []
mnis_562 = []

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

    # filter to only bbgamma
    importance = importance[:, 46:, :]

    # count 4-sigma important probes in each time bin
    importance_threshold = np.mean(importance) + 4 * np.std(importance)
    ctg_counts = np.zeros(48)
    pid_showed_first_predictiveness_for_category = {}
    pid_showed_last_predictiveness_for_category = {}
    for pid in range(successful_probes.shape[0]):
        most_importantce = np.array(importance[pid] > importance_threshold, dtype=np.int8)
        pid_counts = np.array(np.sum(most_importantce, axis=0) > 0, dtype=np.int8)
        ctg_counts += pid_counts
        all_counts += pid_counts

        # identify probes forming 187 - 250 ms peak
        if np.sum(pid_counts[22:24]) > 0:
            mnis_187.append(successful_mnis[pid])

        # identify probes forming 187 - 250 ms peak
        if np.sum(pid_counts[34:36]) > 0:
            mnis_562.append(successful_mnis[pid])

        try:
            first_predictive_at = np.min(np.where(np.sum(most_importantce, axis=0)))
            last_predictive_at = np.max(np.where(np.sum(most_importantce, axis=0)))

            # in-category, first
            probe = tuple(successful_probes[pid])
            if pid_showed_first_predictiveness_for_category.get(probe, None) is None:
                pid_showed_first_predictiveness_for_category[probe] = first_predictive_at
            if first_predictive_at < pid_showed_first_predictiveness_for_category[probe]:
                pid_showed_first_predictiveness_for_category[probe] = first_predictive_at

            # in-category, last
            if pid_showed_last_predictiveness_for_category.get(probe, None) is None:
                pid_showed_last_predictiveness_for_category[probe] = last_predictive_at
            if last_predictive_at > pid_showed_last_predictiveness_for_category[probe]:
                pid_showed_last_predictiveness_for_category[probe] = last_predictive_at

            # globally, first
            if pid_showed_first_predictiveness_globally.get(probe, None) is None:
                pid_showed_first_predictiveness_globally[probe] = first_predictive_at
            if first_predictive_at < pid_showed_first_predictiveness_globally[probe]:
                pid_showed_first_predictiveness_globally[probe] = first_predictive_at

            # globally, last
            if pid_showed_last_predictiveness_globally.get(probe, None) is None:
                pid_showed_last_predictiveness_globally[probe] = last_predictive_at
            if last_predictive_at > pid_showed_last_predictiveness_globally[probe]:
                pid_showed_last_predictiveness_globally[probe] = last_predictive_at

        except:
            print "Probe %d-%d had no importance to show" % (successful_probes[pid][0], successful_probes[pid][1])

    '''
    fig = plt.figure(figsize=(10, 8), dpi=200);
    #ctg_counts -= np.mean(ctg_counts[:15]) + 4 * np.std(ctg_counts[:15]) # normalize: subtract 4 sigma baseline from signal
    #ctg_counts[ctg_counts < 0] = 0.0
    panel_time_profile(ctg_counts, "Number of probes in time that are predictive of %s" % category, ylabel="Number of probes", lines=True)
    plt.savefig('%s/time_count_histogram_%d_%s.png' % (OUTDIR, cid, category), bbox_inches='tight');
    plt.clf();
    plt.close(fig);
    '''

all_first_counts = np.zeros(48)
all_last_counts = np.zeros(48)
for t in pid_showed_first_predictiveness_globally.values():
    all_first_counts[t] += 1
for t in pid_showed_last_predictiveness_globally.values():
    all_last_counts[t] += 1

# for all categories
fig = plt.figure(figsize=(10, 8), dpi=200);
#all_counts -= np.mean(all_counts[:15]) + 4 * np.std(all_counts[:15]) # normalize: subtract 3 sigma baseline from signal
#all_counts[all_counts < 0] = 0.0
panel_time_profile(all_counts, "Number of predictive probes in time over all categories", ylabel="Number of probes", lines=True)
plt.savefig('%s/time_count_histogram_all.png' % OUTDIR, bbox_inches='tight');
plt.clf();
plt.close(fig);

fig = plt.figure(figsize=(10, 8), dpi=200);
panel_time_profile(all_first_counts, "Number of probes that start being predictive at each time moment", ylabel="Number of probes", lines=True)
plt.savefig('%s/first_predictive_time_istogram_all.png' % OUTDIR, bbox_inches='tight');
plt.clf();
plt.close(fig);

fig = plt.figure(figsize=(10, 8), dpi=200);
panel_time_profile(all_last_counts, "Number of probes that stop being predictive at each time moment", ylabel="Number of probes", lines=True)
plt.savefig('%s/last_predictive_time_istogram_all.png' % OUTDIR, bbox_inches='tight');
plt.clf();
plt.close(fig);

# glass brains of the peaks
mnis_187 = np.array(mnis_187)
mnis_562 = np.array(mnis_562)

colors = ['red'] * mnis_187.shape[0] + ['blue'] * mnis_562.shape[0]
duoptych(np.vstack((mnis_187, mnis_562)), np.array(colors), {'blue': 0.3, 'red': 0.3}, ["%s/peaks_187_and_562.png" % OUTDIR])
