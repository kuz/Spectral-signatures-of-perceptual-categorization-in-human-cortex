import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from copy import deepcopy
np.set_printoptions(precision=3, linewidth=300)
import pdb

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
freqsets = ['FT', 'bbgamma', 'lowfreq']
important_clusters = {0: [1, 4, 5, 6],
                      1: [1, 2, 3, 7],
                      2: [1, 3, 6, 10],
                      3: [1, 2, 3, 4],
                      4: [1, 2, 5, 9],
                      5: [1, 2, 8, 9],
                      6: [1, 5, 9, 10],
                      7: [1, 2, 3, 5]}


# load original list of successful probes and their assignation to clusters
successful_probes = {}
successful_mnis = {}
cluster_labels = {}
CLUSTDIR = '../../Outcome/Clustering'
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
for cid in range(len(categories)):
    successful_probes[cid] = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis[cid] = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    cluster_labels[cid] = np.load('%s/%d-%s/successful_probes_to_cluster_labels.npy' % (CLUSTDIR, cid, categories[cid]))

    # drop 0,0,0 electrode if it creeps in
    drop_idx = np.unique(np.where(successful_mnis[cid] == [0, 0, 0])[0])
    successful_probes[cid] = np.delete(successful_probes[cid], drop_idx, 0)
    successful_mnis[cid] = np.delete(successful_mnis[cid], drop_idx, 0)
    cluster_labels[cid] = np.delete(cluster_labels[cid], drop_idx, 0)


# load f1 scores
scores_spc = {}
for freqset in freqsets:
    INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % freqset
    scores_spc[freqset] = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()


# 
for cid, category in enumerate(categories):
    print '%s: (%d probes)' % (category, successful_probes[cid].shape[0])
    
    freqset_f1_scores = {}
    for freqset in freqsets:
        cluster_probe_f1_scores = []

        sum_scores = 0
        num_scores = 0
        for succ_pid, succ_sid in successful_probes[cid]:
            succ_pid -= 1
            try:
                sum_scores += scores_spc[freqset][succ_sid][succ_pid][cid]
                cluster_probe_f1_scores.append(scores_spc[freqset][succ_sid][succ_pid][cid])
                num_scores += 1
            except:
                print "WARNING: cid key missing"
        cluster_predictive_score = sum_scores / float(num_scores)

        freqset_f1_scores[freqset] = cluster_probe_f1_scores
        print '\t%s: %.4f' % (freqset[:2], cluster_predictive_score)
            
    print '\tFT > bb pval = %.6f' % mannwhitneyu(freqset_f1_scores['FT'], freqset_f1_scores['bbgamma'], alternative='greater').pvalue
    print '\tFT > lo pval = %.6f' % mannwhitneyu(freqset_f1_scores['FT'], freqset_f1_scores['lowfreq'], alternative='greater').pvalue
    print '\tbb > lo pval = %.6f' % mannwhitneyu(freqset_f1_scores['bbgamma'], freqset_f1_scores['lowfreq'], alternative='greater').pvalue
    print '\tlo > bb pval = %.6f' % mannwhitneyu(freqset_f1_scores['lowfreq'], freqset_f1_scores['bbgamma'], alternative='greater').pvalue
    print ''

    for cluster_id in important_clusters[cid]:
        print '\tcluster %d (%d probes):' % (cluster_id, successful_probes[cid][cluster_labels[cid] == cluster_id].shape[0])

        freqset_f1_scores = {}
        for freqset in freqsets:
            cluster_probe_f1_scores = []

            # compute cluster's average predictive score (over probes in the cluster)
            sum_scores = 0
            num_scores = 0
            for succ_pid, succ_sid in successful_probes[cid][cluster_labels[cid] == cluster_id]:
                succ_pid -= 1
                sum_scores += scores_spc[freqset][succ_sid][succ_pid][cid]
                cluster_probe_f1_scores.append(scores_spc[freqset][succ_sid][succ_pid][cid])
                num_scores += 1

            if num_scores > 0:
                cluster_predictive_score = sum_scores / float(num_scores)
            else:
                cluster_predictive_score = 0.0

            freqset_f1_scores[freqset] = cluster_probe_f1_scores

            print '\t\t%s: %.4f' % (freqset[:2], cluster_predictive_score)

        print '\t\tFT > bb pval = %.6f' % mannwhitneyu(freqset_f1_scores['FT'], freqset_f1_scores['bbgamma'], alternative='greater').pvalue
        print '\t\tFT > lo pval = %.6f' % mannwhitneyu(freqset_f1_scores['FT'], freqset_f1_scores['lowfreq'], alternative='greater').pvalue
        print '\t\tbb > lo pval = %.6f' % mannwhitneyu(freqset_f1_scores['bbgamma'], freqset_f1_scores['lowfreq'], alternative='greater').pvalue
        print '\t\tlo > bb pval = %.6f' % mannwhitneyu(freqset_f1_scores['lowfreq'], freqset_f1_scores['bbgamma'], alternative='greater').pvalue
        print ''
