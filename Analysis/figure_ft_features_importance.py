import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm

INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
OUTDIR = '../../Outcome/Figures'

fileliest = os.listdir('%s' % INDIR)
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']


# separate plot for each category
for cid, category in enumerate(categories):
    plt.figure(dpi=300);
    
    importance = np.load('%s/%s' % (INDIR, fileliest[cid]))
    importance = np.mean(importance, 0)
    plt.imshow(importance, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');

    plt.colorbar();
    #plt.clim(0, 0.0016);
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
    plt.ylabel('Frequency (Hz)', size=10);
    plt.xlabel('Time (30 ms bin)', size=10);
    plt.title('Importance of spectrotemporal features for "%s"' % categories[cid], size=11);
    plt.savefig('%s/FT_importances_%d_%s.png' % (OUTDIR, cid, category), bbox_inches='tight');
    plt.clf();


# all together
"""
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
plt.title('FT Feature Importances for Different Image Categories', size=12);

plt.savefig('%s/FT_importances.png' % OUTDIR, bbox_inches='tight');
plt.clf();
"""
