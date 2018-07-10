import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as 
from matplotlib.colors import LogNorm

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
OUTDIR = '../../Outcome/Figures'

# functions
def heatmap(d, fname, title, categories, areas_of_interest, absolute=False):
    fig = plt.figure(figsize=(4, 4), dpi=200);
    if absolute:
        plt.imshow(d, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
    else:
        plt.imshow(d, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto', norm=LogNorm(vmin=0.01, vmax=1));
    plt.colorbar();
    #plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
    plt.xticks(np.arange(0, len(areas_of_interest)), areas_of_interest, size=8);
    plt.yticks(np.arange(0, len(categories)), categories, size=8);
    for aid in range(len(areas_of_interest)):
        for cid in range(len(categories)):
            if absolute:
                plt.text(aid, cid, ('%d' % np.round(d[cid, aid] * 100, 0)), va='center', ha='center', size=7)
            else:
                plt.text(aid, cid, ('%d' % np.round(d[cid, aid] * 100, 0) + '%'), va='center', ha='center', size=7)
    #plt.ylabel('Frequency (Hz)', size=10);
    #plt.xlabel('Time (30 ms bin)', size=10);
    plt.title(title, size=11);
    plt.savefig(fname, bbox_inches='tight');
    plt.clf();
    plt.close(fig);

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']

# counte appearances of successfull probes in each area
appearances = {}
total_successful = 0
for cid, category in enumerate(categories):
    appearances[cid] = {}
    successful_areas = np.load('%s/%s' % (INDIR, 'FT_successful_areas_ctg%d.npy' % cid))
    for ba in np.unique(successful_areas):
        appearances[cid][ba] = np.sum(successful_areas == ba)
        total_successful += np.sum(successful_areas == ba)

'''
# see which areas have 5 or more useful probes in them
for cid, category in enumerate(categories):
    for ba in appearances[cid]:
        if appearances[cid][ba] >= 3:
            print "%d: %d" % (ba, appearances[cid][ba])
    print
'''
#areas_of_interest = [17, 18, 19, 20, 37, 6, 21, 22, 36, 48]
areas_of_interest = [17, 18, 19, 20, 37]

# matrix with counts per area
counts = np.zeros((len(categories), len(areas_of_interest)))
for cid, category in enumerate(categories):
    for ba in areas_of_interest:
        counts[cid][areas_of_interest.index(ba)] = appearances[cid].get(ba, 0)

# percentage within category
per_category = counts / counts.sum(axis=1)[:, None]
heatmap(per_category, "%s/heatmap_area_x_category_per_category.png" % OUTDIR, 'Distribution of successful probes within category', categories, areas_of_interest)

# percentage within area
per_area = counts / counts.sum(axis=0)[None, :]
heatmap(per_area, "%s/heatmap_area_x_category_per_area.png" % OUTDIR, 'Distribution of successful probes within area', categories, areas_of_interest)

# percentage out of total
per_total = counts / total_successful
heatmap(per_total, "%s/heatmap_area_x_category_per_total.png" % OUTDIR, 'Distribution of all successful probes', categories, areas_of_interest)

# percentage out of total
heatmap(counts / 100.0, "%s/heatmap_area_x_category_counts.png" % OUTDIR, 'Counts of all successful probes', categories, areas_of_interest, absolute=True)

