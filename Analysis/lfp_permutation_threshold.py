import os
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier

INDIR  = '../../Outcome/Single Probe Classification/LFP/Permutation'
OUTDIR = '../../Outcome'


DATADIR = '../../Data/Intracranial/Processed'
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

plt.figure(dpi=100);
plt.hist(np.ravel(f1_scores[f1_scores > 0.0]), 50, facecolor='green', alpha=0.75);
plt.savefig('%s/Figures/%s' % (OUTDIR, 'LFP_classification_f1_distribution.png'), bbox_inches='tight');
plt.clf();

print 'Out of 11293 permutations the highest F1 score was %.4f' % np.max(f1_scores)