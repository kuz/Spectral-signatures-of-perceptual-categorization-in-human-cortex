import os
import numpy as np
import scipy.io as sio
from scipy.stats import mannwhitneyu
import pdb

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
DATADIR = '../../Data/Intracranial/Processed'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# separate plot for each category
for cid, category in enumerate(categories):

    print '--- %s ---' % category

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

    mi = np.mean(importance, 0)

    # define ROIs
    e_th = mi[ 0: 4, 16:24]
    e_al = mi[ 4:10, 16:24]
    e_be = mi[10:27, 16:24]
    e_lg = mi[27:56, 16:24]
    e_hg = mi[56:  , 16:24]
    
    m_th = mi[ 0: 4, 24:32]
    m_al = mi[ 4:10, 24:32]
    m_be = mi[10:27, 24:32]
    m_lg = mi[27:56, 24:32]
    m_hg = mi[56:  , 24:32] 
    
    l_th = mi[ 0: 4, 32:48]
    l_al = mi[ 4:10, 32:48]
    l_be = mi[10:27, 32:48]
    l_lg = mi[27:56, 32:48]
    l_hg = mi[56:  , 32:48]
    
    times = {'earl': range(16, 24), 'mid': range(24, 32), 'late': range(32, 48)}
    bands = {'theta': range(0, 4), 'alpha': range(4, 10), 'beta ': range(10, 27), 'gamma': range(27, 56), 'bgamma': range(56, 148)}

    #  "high importance of the transient theta activity (theta burst) in all categories"
    #for time, t in times.iteritems():
    #    for band, b in bands.iteritems():
    #        print 'early theta >', time, band, '\t', mannwhitneyu(np.ravel(e_th), np.ravel(mi[b[0]:b[-1]+1, t[0]:t[-1]+1]), alternative='greater')[1] < 8.92857142857143e-06

    # "almost absence of broadband gamma in the control scrambled condition"
    if category == 'scrambled':
        for time, t in times.iteritems():
            for band, b in bands.iteritems():
                print 'bb gamma <', time, band, '\t', mannwhitneyu(np.ravel(np.hstack((e_hg, m_hg, l_hg))), np.ravel(mi[b[0]:b[-1]+1, t[0]:t[-1]+1]), alternative='less')[1] < 8.333333333333333e-05
            