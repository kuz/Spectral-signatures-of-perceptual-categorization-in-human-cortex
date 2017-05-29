import os
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier

INDIR   = '../../Outcome/Single Probe Classification/LFP/Predictions'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR  = '../../Outcome'
featureset = 'LFP_8c_artif_bipolar_BA_responsive'
n_classes = 8

filelist = os.listdir(INDIR)
n_subjects = len(filelist)

# build matrix CLASSES x PROBES x SUBJECTS with F1 scores
print 'Aggregating F1 scores ...'
f1_scores = np.zeros((n_classes, 200, n_subjects))
for sid, filename in enumerate(filelist):
    data = np.load('%s/%s' % (INDIR, filename))
    for pid in data[()].keys():
        f1_scores[:, pid, sid] = f1_score(data[()][pid]['true'], data[()][pid]['pred'], average=None)

# for every category
print 'Retraining RFs on full data for successful probes ...'
for cid in range(n_classes):
    print '\tCategory %d:' % cid

    # for every probe that has high F1 score in the given category
    successful_probes = np.vstack(np.where(f1_scores[cid, :, :] > 0.6)).T
    n_probes = successful_probes.shape[0]
    feature_importances = np.zeros((n_probes, 768))

    for i, (pid, sid) in enumerate(successful_probes):
        print '\t\t%d / %d' % (i + 1, n_probes)

        s = sio.loadmat('%s/%s/%s' % (DATADIR, featureset, filelist[sid].replace('.npy', '.mat')))
        data = s['s']['data'][0][0]
        areas = np.ravel(s['s']['probes'][0][0][0][0][3])
        mni = s['s']['probes'][0][0][0][0][2]
        stimgroups = np.ravel(s['s']['stimgroups'][0][0])
        nprobes = len(areas)

        means = np.mean(data[:, :, 50:230], axis=2).reshape((401, nprobes, 1))
        signal = data - np.broadcast_to(means, (401, nprobes, 768))
        signal = signal[:, :, 256:]

        clf = RandomForestClassifier(n_estimators=1000, n_jobs=4)
        clf.fit(data[:, pid, :], stimgroups);

        feature_counts = np.zeros(clf.estimators_[0].tree_.n_features)
        
        # for each tree in the forest
        for tid in range(clf.n_estimators):
        
            # find leafs that are responsible for the current category
            leafs = np.where(np.logical_and(clf.estimators_[tid].tree_.feature < 0, clf.estimators_[tid].tree_.value[:, 0, cid] > 0.0))[0]
            for leaf in leafs:

                # store which nodes led to that leaf
                path_to_node = []
                while True:
                    parent = np.concatenate((np.where(clf.estimators_[tid].tree_.children_left == leaf)[0], np.where(clf.estimators_[tid].tree_.children_right == leaf)[0]))
                    if len(parent) == 0: break
                    leaf = parent[0]
                    path_to_node.append(leaf)

                # increase counts of the features that are splitting features of the found nodes
                feature_counts[clf.estimators_[tid].tree_.feature[path_to_node]] += 1

        feature_importances[i, :] = feature_counts / np.sum(feature_counts)
        clf = None
    
    # store feature importances
    np.save('%s/Single Probe Classification/LFP/Importances/LFP_feature_importances_ctg%d.npy' % (OUTDIR, cid), feature_importances)

    # plot feature importances
    plt.plot(np.mean(feature_importances, 0));
    plt.savefig('%s/Figures/%s' % (OUTDIR, 'LFP_feature_importances_ctg%d.png' % cid), bbox_inches='tight');
    plt.clf();

