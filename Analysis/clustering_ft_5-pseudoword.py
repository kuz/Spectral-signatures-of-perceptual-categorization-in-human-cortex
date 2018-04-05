import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from scipy.cluster import hierarchy

# parameters
draw_clusters = True
draw_individual = False

INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome'
featureset = 'ft_4hz150_LFP_8c_artif_bipolar_BA_responsive'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
category_ids = [10, 20, 30, 40, 50, 60, 70, 90]
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))
catlist = np.loadtxt('../Preprocessing/stimgroups.txt', dtype=np.int)

def safemkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

# plot definition
def triptych(d, fname, title, lines=True):
    fig = plt.figure(figsize=(8, 6), dpi=300);
    vlim = np.max([np.abs(np.min(d)), np.abs(np.max(d))]) * 2.0
    plt.imshow(d, interpolation='none', origin='lower', cmap=cm.bwr, aspect='auto', vmin=-vlim, vmax=vlim);
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    if lines:
        plt.axvline(x=19, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=24, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axvline(x=32, ymin=0.0, ymax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=4, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=10, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=27, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
        plt.axhline(y=56, xmin=0.0, xmax = 1.0, linewidth=0.5, color='gray', ls='-')
    plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
    plt.ylabel('Frequency (Hz)', size=10);
    plt.xlabel('Time (30 ms bin)', size=10);
    plt.title(title, size=11);
    plt.savefig(fname, bbox_inches='tight');
    plt.clf();
    plt.close(fig);

cid = 5
print '--- Working on "%s" category ---' % categories[cid]

# which stimuli belong to [cid] caterory
catimages = catlist == category_ids[cid]

# load feature importances
importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))

# load list of successfull probes in [cid] category
successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
print "Number of successful probes: %d" % (successful_probes.shape[0])

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

# cluster using ward linkage was better for this case as it better covers class diversity
Z = hierarchy.linkage(X, 'complete', 'cosine');
cluster_labels = hierarchy.fcluster(Z, 18, criterion='maxclust')

# manually merge clusters based on observations
cluster_labels[cluster_labels == 18] = 1
cluster_labels[cluster_labels == 10] = 9

successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
successful_areas = np.load('%s/%s' % (INDIR, 'FT_successful_areas_ctg%d.npy' % cid))

# store mapping of successful probes to clusters
safemkdir('%s/Clustering/%d-%s' % (OUTDIR, cid, categories[cid]))
np.save('%s/Clustering/%d-%s/successful_probes_to_cluster_labels.npy' % (OUTDIR, cid, categories[cid]), cluster_labels)
np.save('%s/Clustering/%d-%s/important_activity_patterns.npy' % (OUTDIR, cid, categories[cid]), important_activity_patterns)

# print out BAs per cluster
for cluster_id in np.unique(cluster_labels):
    print 'Cluster %d:' % cluster_id, np.unique(successful_areas[cluster_labels == cluster_id])

if draw_clusters:

    # draw cluster means
    print "Drawing cluster means..."
    for cluster_id in np.unique(cluster_labels):
        safemkdir('%s/Figures/Clustering/%d-%s' % (OUTDIR, cid, categories[cid]))
        triptych(np.mean(important_activity_patterns[cluster_labels == cluster_id], axis=0),
                 '%s/Figures/Clustering/%d-%s/Cluster-%d' % (OUTDIR, cid, categories[cid], cluster_id),
                 'Cluster %d mean on "%s" category' % (cluster_id, categories[cid]))

if draw_individual:

    # draw individual samples grouped into clusters
    print "Particular instances..."
    for i, (pid, sid) in enumerate(successful_probes):
        print "\t%d/%d" % (i + 1, successful_probes.shape[0])
        safemkdir('%s/Figures/Clustering/%d-%s/C%d' % (OUTDIR, cid, categories[cid], cluster_labels[i]))
        triptych(important_activity_patterns[i],
                 '%s/Figures/Clustering/%d-%s/C%d/s%d-p%d' % (OUTDIR, cid, categories[cid], cluster_labels[i], sid, pid),
                 'Response of succ. probe s%d-p%d to "%s"' % (sid, pid, categories[cid]))
