import os
import numpy as np
from matplotlib import pylab as plt
from functions_plotting import panel_time_profile

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures/time_profile'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
areas = [17, 18, 19, 37, 20]

# separate plot for each category
for cid, category in enumerate(categories):

    print 'Drawing "%s" ...' % category
    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    successful_areas = np.load('%s/%s' % (INDIR, 'FT_successful_areas_ctg%d.npy' % cid))

    # drop 0,0,0 electrode if it creeps in
    drop_idx = np.unique(np.where(successful_mnis == [0, 0, 0])[0])
    importance = np.delete(importance, drop_idx, 0)
    successful_probes = np.delete(successful_probes, drop_idx, 0)
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)
    successful_areas = np.delete(successful_areas, drop_idx, 0)

    # Mean over each BA
    for ba in areas:
        if importance[successful_areas == ba, :, :].shape[0] > 0:
            fig = plt.figure(figsize=(10, 8), dpi=300);
            title = 'Temporal profile of importance for "%s"\nmean over %d probes in BA%d' % (categories[cid], np.sum(successful_areas == ba), ba)
            panel_time_profile(np.sum(np.mean(importance[successful_areas == ba, :, :], 0), 0), title, True)
            plt.savefig('%s/time_profile_%d_%s_BA%d.png' % (OUTDIR, cid, category, ba), bbox_inches='tight');
            plt.clf();
            plt.close(fig);

    
        
