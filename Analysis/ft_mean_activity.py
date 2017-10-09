import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures/Mean ratio'

categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
category_ids = [10, 20, 30, 40, 50, 60, 70, 90]
stimgroups = np.loadtxt('../Preprocessing/stimgroups.txt').astype('int')
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))

try:
    os.mkdir('%s' % OUTDIR)
except:
    pass

for cid in range(len(categories)):
    category = categories[cid]
    print 'Drawing %s' % category

    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))

    mask = np.mean(importance, 0)
    mask[mask < (np.mean(mask) + np.std(mask))] = 0.0
    mask[mask > 0.0] = 1.0

    try:
        os.mkdir('%s/%d_%s' % (OUTDIR, cid, category))
    except:
        pass

    for i in range(importance.shape[0]):
        print "\t%d/%d" % (i + 1, importance.shape[0])

        (pid, sid) = successful_probes[i]
        sname = subjlist[sid].replace('.npy', '')
        
        #s = sio.loadmat('%s/%s/%s' % (DATADIR, 'ft_4hz150_LFP_8c_artif_bipolar_BA_responsive', '%s-%d.mat' % (sname, pid)))
        #d = s['ft'][:, :, :] * mask
        #d = s['ft'][:, :, :]
        
        data = np.load('%s/%s/%s' % (DATADIR, 'normalized_ft_4hz150_LFP_8c_artif_bipolar_BA_responsive', '%s-%d.npy' % (sname, pid)))
        #d = data * mask
        d = data
        
        m = np.mean(d[stimgroups == category_ids[cid], :, :], 0)
        print np.min(m), np.max(m)
        fig = plt.figure(figsize=(8, 6), dpi=300);
        plt.imshow(m, interpolation='none', origin='lower', cmap=cm.bwr, aspect='auto');
        #plt.imshow(m, interpolation='none', origin='lower', cmap=cm.jet, aspect='auto');
        plt.clim(-4.0, 4.0)
        plt.colorbar();
        plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
        plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
        plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
        plt.ylabel('Frequency (Hz)', size=10);
        plt.xlabel('Time (30 ms bin)', size=10);
        plt.title('mean ratio %s-%d %s' % (sname, pid, category), size=11);
        plt.savefig('%s/%d_%s/mean_ratio_%s-%d-%d_%s' % (OUTDIR, cid, category, sname, sid, pid, category), bbox_inches='tight');
        #plt.savefig('%s/%d_%s/mean_ratio_%s-%d-%d_%s' % (OUTDIR, cid, category, sname, sid, pid, category), bbox_inches='tight');
        plt.clf();
        plt.close(fig);

# clustering
