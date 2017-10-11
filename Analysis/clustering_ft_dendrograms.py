import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from scipy.cluster import hierarchy

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'
featureset = 'ft_4hz150_LFP_8c_artif_bipolar_BA_responsive'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
category_ids = [10, 20, 30, 40, 50, 60, 70, 90]
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))
catlist = np.loadtxt('../Preprocessing/stimgroups.txt', dtype=np.int)

for cid in range(len(categories)):

    print '--- Working on "%s" category ---' % categories[cid]

    # which stimuli belong to [cid] caterory
    catimages = catlist == category_ids[cid]

    # load feature importances
    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    
    # load list of successfull probes in [cid] category
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))

    # extract activity (baseline normalized power) of the activyt of succesful probes masked by feature importance
    important_activity_patterns = np.zeros(importance.shape)
    for i, (pid, sid) in enumerate(successful_probes):
        sname = subjlist[sid].split('.')[0]
        importance_threshold = np.mean(importance[i]) + np.std(importance[i])
        most_important_features = importance[i] > importance_threshold
        ft = np.load('%s/normalized_%s/%s-%d.npy' % (DATADIR, featureset, sname, int(pid)))
        important_activity_patterns[i] = np.mean(ft[catimages, :, :], axis=0) * most_important_features

    # binarize importance to make it easier for clustering
    X = important_activity_patterns.reshape(successful_probes.shape[0], 7008).copy()
    X[X > 0.0] =  2.0
    X[X < 0.0] = -2.0

    # cluster using complete cosine linkage
    Z = hierarchy.linkage(X, 'complete', 'cosine');
    fig = plt.figure();
    dn = hierarchy.dendrogram(Z);
    plt.savefig('%s/Clustering/dendrogram-%d-%s-complete-cosine.png' % (OUTDIR, cid, categories[cid]), bbox_inches='tight');
    plt.close(fig)

    # cluster using ward linkage
    Z = hierarchy.linkage(X, 'ward', 'euclidean');
    fig = plt.figure();
    dn = hierarchy.dendrogram(Z);
    plt.savefig('%s/Clustering/dendrogram-%d-%s-ward.png' % (OUTDIR, cid, categories[cid]), bbox_inches='tight');
    plt.close(fig)
