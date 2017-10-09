import os
import numpy as np
import scipy.io as sio

# parameters
INDIR = '../../Outcome/Single Probe Classification/FT/Importances'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Single Probe Classification/FT/Importances'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/FT/Predictions'))
n_subjects = len(subjlist)

# separate plot for each category
for cid, category in enumerate(categories):

    print 'Computing "%s" ...' % category

    importance = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid))
    successful_probes = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' %  cid))
    successful_mnis = []
    successful_areas = []

    # Plot each probe's importances
    for i in range(importance.shape[0]):
        print '%d / %d' % (i + 1, importance.shape[0])

        # load MNI information
        (pid, sid) = successful_probes[i]
        pid = pid - 1
        s = sio.loadmat('%s/%s/%s' % (DATADIR, 'LFP_8c_artif_bipolar_BA_responsive', subjlist[sid].replace('.npy', '.mat')))
        areas = np.ravel(s['s']['probes'][0][0][0][0][3])
        mni = s['s']['probes'][0][0][0][0][2]
        successful_mnis.append(mni[pid])
        successful_areas.append(areas[pid])

    successful_mnis = np.array(successful_mnis)
    successful_areas = np.array(successful_areas)

    np.save('%s/FT_successful_mnis_ctg%d.npy' % (OUTDIR, cid), successful_mnis)
    np.save('%s/FT_successful_areas_ctg%d.npy' % (OUTDIR, cid), successful_areas)
