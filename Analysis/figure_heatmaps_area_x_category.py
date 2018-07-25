import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
OUTDIR = '../../Outcome/Figures'

# functions
def heatmap(data, counts, fname, title, categories, areas_of_interest):
    fig = plt.figure(figsize=(4, 4), dpi=200);
    plt.imshow(data, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto', norm=LogNorm(vmin=0.01, vmax=1));
    plt.colorbar();
    plt.xticks(np.arange(0, len(areas_of_interest)), areas_of_interest, size=8);
    plt.yticks(np.arange(0, len(categories)), categories, size=8);
    for aid in range(len(areas_of_interest)):
        for cid in range(len(categories)):
            plt.text(aid, cid, (str(int(np.round(data[cid, aid] * 100, 0))) + '% (' + str(int(counts[cid, aid])) + ')'), va='center', ha='center', size=6)
    plt.title(title, size=11);
    plt.savefig(fname, bbox_inches='tight');
    plt.clf();
    plt.close(fig);

# lists
categories = ['house', 'face', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']

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
areas_of_interest = [17, 18, 19, 37, 20]

# matrix with counts per area
counts = np.zeros((len(categories), len(areas_of_interest)))
for cid, category in enumerate(categories):
    for ba in areas_of_interest:
        counts[cid][areas_of_interest.index(ba)] = appearances[cid].get(ba, 0)

# percentage within category
per_category = counts / counts.sum(axis=1)[:, None]
heatmap(per_category, counts, "%s/heatmap_area_x_category_per_category.png" % OUTDIR, 'Distribution of successful probes within category', categories, areas_of_interest)

# percentage within area
per_area = counts / counts.sum(axis=0)[None, :]
heatmap(per_area, counts, "%s/heatmap_area_x_category_per_area.png" % OUTDIR, 'Distribution of successful probes within area', categories, areas_of_interest)

'''
# percentage out of total
per_total = counts / total_successful
heatmap(per_total, "%s/heatmap_area_x_category_per_total.png" % OUTDIR, 'Distribution of all successful probes', categories, areas_of_interest)
'''

'''
# percentage out of total
heatmap(counts / 100.0, "%s/heatmap_area_x_category_counts.png" % OUTDIR, 'Counts of all successful probes', categories, areas_of_interest, absolute=True)
'''
