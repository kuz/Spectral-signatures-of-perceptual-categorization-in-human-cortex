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
fig = plt.figure(figsize=(10, 8), dpi=300);
title = 'Distribution of probe polypredictiveness'

plt.bar(range(8), counts[1:], color="blue");
plt.xticks(range(8), np.arange(1, 9), size=11, rotation=90);
plt.ylabel('Number of probes', size=16);
plt.xlabel('Number of categories a probe predicts', size=16);
plt.title(title, size=16);

plt.savefig('%s/predictiveness_of_probes.png' % OUTDIR, bbox_inches='tight');
plt.clf();
plt.close(fig);
