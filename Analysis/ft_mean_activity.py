import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from functions_plotting import panel_activity, panel_importance
from functions_helpers import safemkdir
import pdb

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
category_ids = [10, 20, 30, 40, 50, 60, 70, 90]
stimgroups = np.loadtxt('../Preprocessing/stimgroups.txt').astype('int')
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))

safemkdir('%s' % OUTDIR)
safemkdir('%s/Importance' % OUTDIR)
safemkdir('%s/Mean ratio' % OUTDIR)
safemkdir('%s/Mean ratio contour' % OUTDIR)
safemkdir('%s/Mean ratio masked' % OUTDIR)
safemkdir('%s/Mean ratio masked contour' % OUTDIR)

for cid in range(len(categories)):
    category = categories[cid]
    print 'Drawing %s' % category

    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))

    #category_mask = np.mean(importance, 0)
    #category_mask[category_mask < (np.mean(category_mask) + np.std(category_mask))] = 0.0
    #category_mask[category_mask > 0.0] = 1.0 

    safemkdir('%s/Importance/%d_%s' % (OUTDIR, cid, category))
    safemkdir('%s/Mean ratio/%d_%s' % (OUTDIR, cid, category))
    safemkdir('%s/Mean ratio contour/%d_%s' % (OUTDIR, cid, category))
    safemkdir('%s/Mean ratio masked/%d_%s' % (OUTDIR, cid, category))
    safemkdir('%s/Mean ratio masked contour/%d_%s' % (OUTDIR, cid, category))

    for i in range(importance.shape[0]):
        print "\t%d/%d" % (i + 1, importance.shape[0])

        (pid, sid) = successful_probes[i]
        sname = subjlist[sid].replace('.npy', '')

        data = np.load('%s/%s/%s' % (DATADIR, 'normalized_ft_4hz150_LFP_8c_artif_bipolar_BA_responsive', '%s-%d.npy' % (sname, pid)))
        activity = np.mean(data[stimgroups == category_ids[cid], :, :], 0)

        importance_threshold = np.mean(importance[i]) + np.std(importance[i])
        mask = importance[i] > importance_threshold

        masked_data = data * mask
        masked_activity = np.mean(masked_data[stimgroups == category_ids[cid], :, :], 0)
        
        # importance
        fig = plt.figure(figsize=(10, 8), dpi=200);
        panel_importance(importance[i], '%s-%d %s' % (sname, pid - 1, category), True)
        plt.savefig('%s/Importance/%d_%s/importance_%s-%d-%d_%s' % (OUTDIR, cid, category, sname, sid, pid - 1, category), bbox_inches='tight');
        plt.clf();
        plt.close(fig);  

        # mean ratio
        fig = plt.figure(figsize=(10, 8), dpi=200);
        panel_activity(activity, '%s-%d %s' % (sname, pid - 1, category), True)
        plt.savefig('%s/Mean ratio/%d_%s/mean_ratio_%s-%d-%d_%s' % (OUTDIR, cid, category, sname, sid, pid - 1, category), bbox_inches='tight');
        plt.clf();
        plt.close(fig);        

        # mean ratio contour
        fig = plt.figure(figsize=(10, 8), dpi=200);
        panel_activity(activity, '%s-%d %s' % (sname, pid - 1, category), True, mask)
        plt.savefig('%s/Mean ratio contour/%d_%s/mean_ratio_countour_%s-%d-%d_%s' % (OUTDIR, cid, category, sname, sid, pid - 1, category), bbox_inches='tight');
        plt.clf();
        plt.close(fig);

        # masked
        fig = plt.figure(figsize=(10, 8), dpi=200);
        panel_activity(masked_activity, '%s-%d %s' % (sname, pid - 1, category), True)
        plt.savefig('%s/Mean ratio masked/%d_%s/mean_masked_ratio_%s-%d-%d_%s' % (OUTDIR, cid, category, sname, sid, pid - 1, category), bbox_inches='tight');
        plt.clf();
        plt.close(fig);

        # masked contour
        fig = plt.figure(figsize=(10, 8), dpi=200);
        panel_activity(masked_activity, '%s-%d %s' % (sname, pid - 1, category), True, mask)
        plt.savefig('%s/Mean ratio masked contour/%d_%s/mean_masked_ratio_contour_%s-%d-%d_%s' % (OUTDIR, cid, category, sname, sid, pid - 1, category), bbox_inches='tight');
        plt.clf();
        plt.close(fig);
