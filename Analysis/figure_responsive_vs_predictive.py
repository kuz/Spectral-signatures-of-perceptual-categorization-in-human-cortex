"""

To setup Freesurfer run
$ export FREESURFER_HOME=/Applications/freesurfer
$ source $FREESURFER_HOME/SetUpFreeSurfer.sh

"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import matplotlib.cm as cm
import scipy.io as sio
from surfer import Brain
import pdb

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures'

# surfer parameters
subject_id = "fsaverage"

# lists
#categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
categories = ['house']
category_codes = ['10', '20', '30', '40', '50', '60', '70', '90']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))

def duoptych(foci, foci_colors, filenames):
    
    fig = plt.figure(figsize=(16, 8), dpi=200);

    # 3D mesh
    ax1 = plt.subplot2grid((1, 2), (0, 0), colspan=1, rowspan=1)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        if color == 'blue':
            brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.15, color=color);
        else:
            brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.3, color=color);
    brain.show_view('m');
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlabel('MNI Y', size=16);
    plt.xticks(np.arange(0, 800, 20), np.asarray(np.arange(-75, 106, 4.5), dtype='int'), size=10, rotation=90);
    plt.ylabel('MNI Z', size=16);
    plt.yticks(np.arange(0, 800, 20), np.asarray(np.arange(96, -87, -4.372), dtype='int'), size=10);
    plt.xlim(0, 800);
    plt.ylim(800, 0);
    ax1.text(10, 39, '1', fontsize=20)

    # 3D mesh
    ax1 = plt.subplot2grid((1, 2), (0, 1), colspan=1, rowspan=1)
    brain = Brain(subject_id, "both", "pial", cortex='ivory', alpha=0.1, background='white');
    for color in np.unique(foci_colors):
        if color == 'blue':
            brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.15, color=color);
        else:
            brain.add_foci(foci[foci_colors==color, :], hemi='rh', scale_factor=0.3, color=color);
    brain.show_view(view=dict(azimuth=0.0, elevation=0), roll=90);
    pic = brain.screenshot()
    plt.imshow(pic);
    plt.xlabel('MNI Y', size=16);
    plt.xticks(np.arange(0, 800, 20), np.asarray(np.arange(-75, 106, 4.5), dtype='int'), size=10, rotation=90);
    plt.ylabel('MNI X', size=16);
    plt.yticks(np.arange(0, 800, 20), np.asarray(np.arange(-94.5, 93.5, 4.7), dtype='int'), size=10);
    plt.xlim(0, 800);
    plt.ylim(800, 0);
    ax1.text(10, 39, '2', fontsize=20)

    # store the figure
    for filename in filenames:
        plt.savefig(filename, bbox_inches='tight');
    plt.clf();
    plt.close(fig);

# separate plot for each category
for cid, category in enumerate(categories):

    print 'Drawing "%s" ...' % category

    # load predictive mnis
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    drop_idx = np.unique(np.where(successful_mnis == [0, 0, 0])[0])
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)

    # collect responsive probes from all subjects
    foci = np.empty((0, 3))
    colors = []
    for sid in range(len(subjlist)):
        s = sio.loadmat('%s/%s%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive_cat', category_codes[cid], subjlist[sid].replace('.npy', '.mat')))
        responsive_mnis = s['s']['probes'][0][0][0][0][2]
        if responsive_mnis.shape[0] > 0:
            foci = np.vstack((foci, responsive_mnis))
            colors += ['blue'] * responsive_mnis.shape[0]

    # add predictive probes to the list
    colors += ['green'] * successful_mnis.shape[0]
    foci = np.vstack((foci, successful_mnis))

    # plot
    duoptych(foci, np.array(colors), ["%s/responsive_vs_predictive_%s.png" % (OUTDIR, category)])

