import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from surfer import Brain

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

# surfer parameters
subject_id = "fsaverage"
subjects_dir = os.environ["SUBJECTS_DIR"]

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))
n_subjects = len(subjlist)

# separate plot for each category
for cid, category in enumerate(categories):

    print 'Drawing "%s" ...' % category

    subdir = 'FT_importances_%d_%s' % (cid, category)
    try:
        os.mkdir('%s/%s' % (OUTDIR, subdir))
    except:
        pass

    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid))
    successful_mnis = []

    # Plot each probe's importances
    for i in range(importance.shape[0]):
        print '%d / %d' % (i + 1, importance.shape[0])

        # load MNI information
        (pid, sid) = successful_probes[i]
        pid = pid - 1
        s = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', subjlist[sid].replace('.npy', '.mat')))
        areas = np.ravel(s['s']['probes'][0][0][0][0][3])
        mni = s['s']['probes'][0][0][0][0][2]

        # figure
        plt.figure(figsize=(16, 6), dpi=300);

        # feature importances
        plt.subplot(1, 2, 1);
        plt.imshow(importance[i, :, :], interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
        plt.colorbar();
        plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--');
        plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
        plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
        plt.ylabel('Frequency (Hz)', size=10);
        plt.xlabel('Time (30 ms bin)', size=10);
        plt.title('Importance of spectrotemporal features for "%s"\nsID: %d   pID: %d   BA: %d' % (categories[cid], sid, pid, areas[pid]), size=11);

        # 3D mesh
        plt.subplot(1, 2, 2);
        brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
        brain.show_view(view=dict(azimuth=152.88, elevation=65.94), roll=101.58);
        brain.add_foci(mni[pid], hemi='rh', scale_factor=1.0, color='blue');
        successful_mnis.append(mni[pid])
        pic = brain.screenshot()
        plt.imshow(pic);
        
        plt.savefig('%s/%s/FT_importances_%d_%s_pid-%d.png' % (OUTDIR, subdir, cid, category, i), bbox_inches='tight');
        plt.clf();

    # Mean over all probe's importances
    plt.figure(figsize=(16, 6), dpi=300);
    
    plt.subplot(1, 2, 1);
    importance = np.mean(importance, 0)
    plt.imshow(importance, interpolation='none', origin='lower', cmap=cm.Blues, aspect='auto');
    plt.colorbar();
    plt.axvline(x=16, ymin=0.0, ymax = 1.0, linewidth=1.0, color='r', ls='--')
    plt.xticks(np.arange(0, 48), np.asarray((np.arange(0, 769, 16) - 256) / 512.0 * 1000, dtype='int'), size=5, rotation=90);
    plt.yticks(np.arange(0, 146, 5), np.arange(4, 150, 5), size=5);
    plt.ylabel('Frequency (Hz)', size=10);
    plt.xlabel('Time (30 ms bin)', size=10);
    plt.title('Importance of spectrotemporal features for "%s"' % categories[cid], size=11);

    # 3D mesh
    plt.subplot(1, 2, 2);
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.4, background='white');
    brain.show_view(view=dict(azimuth=152.88, elevation=65.94), roll=101.58);
    brain.add_foci(successful_mnis, hemi='rh', scale_factor=0.5, color='blue');
    pic = brain.screenshot()
    plt.imshow(pic);

    plt.savefig('%s/%s/FT_importances_%d_%s_MEAN.png' % (OUTDIR, subdir, cid, category), bbox_inches='tight');
    plt.clf();
