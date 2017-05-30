import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm

INDIR = '../../Outcome/Single Probe Classification/LFP/Importances'
OUTDIR = '../../Outcome/Figures'

fileliest = os.listdir('%s' % INDIR)
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

# separate plot for each category
for cid, category in enumerate(categories):
    plt.figure(dpi=300);
    
    importance = np.load('%s/%s' % (INDIR, fileliest[cid]))
    importance = np.mean(importance, 0)
    plt.plot(importance, linewidth=0.5);
    plt.ylim(0, 0.007);

    plt.legend([categories[cid]]);
    plt.axvline(x=256, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 769, 16), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
    plt.xlabel('Feature (time in ms)', size=10);
    plt.ylabel('Feature importance', size=10);
    plt.title('LFP Feature Importances for Different Image Categories', size=12);

    plt.savefig('%s/LFP_importances_%d_%s_.png' % (OUTDIR, cid, category), bbox_inches='tight');
    plt.clf();

# all together
plt.figure(dpi=300);
for cid, category in enumerate(categories):
    importance = np.load('%s/%s' % (INDIR, fileliest[cid]))
    importance = np.mean(importance, 0)
    plt.plot(importance, linewidth=0.5, ls=linestyles[cid]);

plt.legend(categories);
plt.axvline(x=256, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
plt.xticks(np.arange(0, 769, 16), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
plt.xlabel('Feature (time in ms)', size=10);
plt.ylabel('Feature importance', size=10);
plt.title('LFP Feature Importances for Different Image Categories', size=12);

plt.savefig('%s/LFP_importances.png' % OUTDIR, bbox_inches='tight');
plt.clf();