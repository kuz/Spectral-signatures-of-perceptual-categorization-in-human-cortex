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
from functions_plotting import duoptych

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
            colors += ['green'] * responsive_mnis.shape[0]

    # add predictive probes to the list
    colors += ['blue'] * successful_mnis.shape[0]
    foci = np.vstack((foci, successful_mnis))

    # plot
    duoptych(foci, np.array(colors), {'blue': 0.3, 'green': 0.15}, ["%s/responsive_vs_predictive_%s.png" % (OUTDIR, category)])
