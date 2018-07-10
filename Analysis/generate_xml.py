import numpy as np
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
csvresults = []
for cid, category in enumerate(categories):
    print '%s: (%d probes)' % (category, successful_probes[cid].shape[0])
    
    for cnum, cluster_id in enumerate(important_clusters[cid]):
        print '\tcluster %d (%d probes):' % (cluster_id, successful_probes[cid][cluster_labels[cid] == cluster_id].shape[0])
        
        mnis = successful_mnis[cid][cluster_labels[cid] == cluster_id]
        for i, (succ_pid, succ_sid) in enumerate(successful_probes[cid][cluster_labels[cid] == cluster_id]):
            succ_pid -= 1
            record = [category, cnum + 1, succ_sid, succ_pid,
                      mnis[i][0], mnis[i][1], mnis[i][2],
                      scores_spc['FT'][succ_sid][succ_pid][cid],
                      scores_spc['bbgamma'][succ_sid][succ_pid][cid],
                      scores_spc['lowfreq'][succ_sid][succ_pid][cid]]
            csvresults.append(record)

with open('../../Outcome/all_predictive.csv', 'w') as fh:
    fh.write('Category, Cluster, Subject, Probe, MNI X, MNI Y, MNI Z, Predictive FT, Predictive BB, Predictive LO\n')
    for r in csvresults:
        fh.write('%s, %d, %d, %d, %.3f, %.3f, %.3f, %.4f, %.4f, %.4f\n' % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))
