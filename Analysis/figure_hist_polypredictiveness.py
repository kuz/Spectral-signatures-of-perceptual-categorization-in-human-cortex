import os
import numpy as np
from matplotlib import pylab as plt
import scipy.io as sio
import pdb

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
OUTDIR = '../../Outcome/Figures/polymono'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# aggregate how many probes are predictive of how many categories
counts = np.zeros(9, dtype=np.int32)
for sid in scores_spc.keys():
    for pid in scores_spc[sid].keys():
        counts[len(scores_spc[sid][pid])] += 1

# plot the histogram
fig = plt.figure(figsize=(4, 8), dpi=300);

plt.barh(range(8), counts[1:], color="green");
plt.xlabel('Number of probes', size=16);
plt.ylabel('Number of categories a probe predicts', size=16);
plt.yticks(range(8), np.arange(1, 9), size=14);
plt.xticks([0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400],
           [0, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400], size=12, rotation=90);
for i, v in enumerate(counts[1:]):
    if i == 0:
        plt.gca().text(v - 65, i + 0.1, str(counts[1:][i]), color='black', size=14)
    else:
        plt.gca().text(v + 12, i + 0.1, str(counts[1:][i]), color='black', size=14)
plt.gca().invert_yaxis()
plt.title('Probe polypredictiveness', size=18);

plt.savefig('%s/predictiveness_of_probes.png' % OUTDIR, bbox_inches='tight');
plt.clf();
plt.close(fig);


# pairwise category network overlap
n_categories = len(categories)
overlap = np.zeros((n_categories, n_categories))
for sid in scores_spc.keys():
    for pid in scores_spc[sid].keys():

        if len(scores_spc[sid][pid]) > 1:

            for cid_a in range(n_categories):
                for cid_b in range(n_categories):

                    if scores_spc[sid][pid].get(cid_a, None) is not None and scores_spc[sid][pid].get(cid_b, None) is not None:
                        overlap[cid_a, cid_b] += 1

print overlap


