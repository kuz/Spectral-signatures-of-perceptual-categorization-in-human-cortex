import os
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier

# cmd arguments
parser = argparse.ArgumentParser(description='Build RF models of successful probes and store feature importances')
parser.add_argument('-c', '--cid', dest='cid', type=int, required=True, help='Image category (0 to 7)')
args = parser.parse_args()
cid = int(args.cid)

# parameters
INDIR   = '../../Outcome/Single Probe Classification/FT/Predictions'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR  = '../../Outcome'
featureset = 'ft_4hz150_LFP_8c_artif_bipolar_BA_responsive'
n_classes = 8
n_freqs = 146
threshold = 0.390278 # estimated as 99.999 percentile over permutation F1 scores

filelist = sorted(os.listdir(INDIR))
n_subjects = len(filelist)

# build matrix CLASSES x PROBES x SUBJECTS with F1 scores
print 'Aggregating F1 scores ...'
f1_scores = np.zeros((n_classes, 200, n_subjects))
for sid, filename in enumerate(filelist):
    data = np.load('%s/%s' % (INDIR, filename))
    for pid in data[()].keys():
        f1_scores[:, pid, sid] = f1_score(data[()][pid]['true'], data[()][pid]['pred'], average=None)

print 'Retraining RFs on full data for successful probes ...'
print 'Category %d:' % cid

# for every probe that has high F1 score in the given category
successful_probes = np.vstack(np.where(f1_scores[cid, :, :] > threshold)).T
n_probes = successful_probes.shape[0]
feature_importances = np.zeros((n_probes, 146, 48))

for i, (pid, sid) in enumerate(successful_probes):
    print '\t%d / %d' % (i + 1, n_probes)

    # load subject data
    s = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', filelist[sid].replace('.npy', '.mat')))
    areas = np.ravel(s['s']['probes'][0][0][0][0][3])
    mni = s['s']['probes'][0][0][0][0][2]
    stimgroups = np.ravel(s['s']['stimgroups'][0][0])
    nprobes = len(areas)

    # load spectral responses
    ft = sio.loadmat('%s/%s/%s-%d.mat' % (DATADIR, featureset, filelist[sid].replace('.npy', ''), pid))
    
    # normalize by baseline
    baseline = ft['ft'][:, :, 0:13]
    means = np.mean(baseline, axis=2).reshape((401, n_freqs, 1))
    signal = ft['ft'] / np.broadcast_to(means, (401, n_freqs, 48))
    signal[np.isnan(signal)] = 0.0

    # reshape for RF
    signal = signal.reshape(signal.shape[0], signal.shape[1] * signal.shape[2])

    # train a classifier
    clf = RandomForestClassifier(n_estimators=3000, n_jobs=16)
    clf.fit(signal, stimgroups);
    
    # for each tree in the forest
    forest_importances = np.zeros(clf.estimators_[0].tree_.n_features)
    for tid in range(clf.n_estimators):

        tree = clf.estimators_[tid].tree_
        tree_importances = np.zeros(tree.n_features)

        # find leafs that are responsible for the current category
        leafs = np.where(np.logical_and(tree.feature < 0, tree.value[:, 0, cid] > 0.0))[0]
        for leaf in leafs:

            # store which nodes led to that leaf
            path_to_node = []
            while True:
                parent = np.concatenate((np.where(tree.children_left == leaf)[0], np.where(tree.children_right == leaf)[0]))
                if len(parent) == 0: break
                leaf = parent[0]
                path_to_node.append(leaf)

            for nid in path_to_node:
                fid = tree.feature[nid]
                left = tree.children_left[nid]
                right = tree.children_right[nid]
                tree_importances[fid] += (tree.weighted_n_node_samples[nid] * tree.impurity[nid] -
                                          tree.weighted_n_node_samples[left] * tree.impurity[left] - 
                                          tree.weighted_n_node_samples[right] * tree.impurity[right])
        
        forest_importances += tree_importances

    feature_importances[i, :, :] = (forest_importances / np.sum(forest_importances)).reshape(146, 48)
    clf = None

# store feature importances
np.save('%s/Single Probe Classification/FT/Importances/FT_feature_importances_ctg%d.npy' % (OUTDIR, cid), feature_importances)
np.save('%s/Single Probe Classification/FT/Importances/FT_successful_probes_ctg%d.npy' % (OUTDIR, cid), successful_probes)

