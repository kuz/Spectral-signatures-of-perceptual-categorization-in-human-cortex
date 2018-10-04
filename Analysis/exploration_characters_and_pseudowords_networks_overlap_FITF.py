import os
import numpy as np
from functions_plotting import triptych_importance, triptych_fitfdiff
from functions_helpers import safemkdir
from matplotlib import pylab as plt
import scipy.io as sio
from scipy.stats import mannwhitneyu
import pdb

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET
CLUSTDIR = '../../Outcome/Clustering'
DATADIR = '../../Data/Intracranial/Processed'
OUTDIR = '../../Outcome/Figures/FITF_diff'

# lists
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']

# load data
importance_char = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % 6))
successful_probes_char = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % 6))
successful_mnis_char = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % 6))
drop_idx_char = np.unique(np.where(successful_mnis_char == [0, 0, 0])[0])
importance_char = np.delete(importance_char, drop_idx_char, 0)
successful_probes_char = np.delete(successful_probes_char, drop_idx_char, 0)
successful_mnis_char = np.delete(successful_mnis_char, drop_idx_char, 0)

importance_pseu = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % 5))
successful_probes_pseu = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % 5))
successful_mnis_pseu = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % 5))
drop_idx_pseu = np.unique(np.where(successful_mnis_pseu == [0, 0, 0])[0])
importance_pseu = np.delete(importance_pseu, drop_idx_pseu, 0)
successful_probes_pseu = np.delete(successful_probes_pseu, drop_idx_pseu, 0)
successful_mnis_pseu = np.delete(successful_mnis_pseu, drop_idx_pseu, 0)

# split into 3 groups
in_both = np.empty((0, 3))
only_in_char = np.empty((0, 3))
only_in_pseu = np.empty((0, 3))

importance_new_both = np.empty((0, 146, 48))
importance_new_char = np.empty((0, 146, 48))
importance_new_pseu = np.empty((0, 146, 48))

successful_mnis_char_list = successful_mnis_char.tolist()
successful_mnis_pseu_list = successful_mnis_pseu.tolist()
successful_mnis_unique = np.unique(np.vstack((successful_mnis_char, successful_mnis_pseu)), axis=0).tolist()
for mni in successful_mnis_unique:
    if (mni in successful_mnis_char_list) and (mni in successful_mnis_pseu_list):
        in_both = np.vstack((in_both, mni))
        id_in_char = np.where(successful_mnis_char == mni)[0][0]
        id_in_pseu = np.where(successful_mnis_pseu == mni)[0][0]
        importance_new_both = np.vstack((importance_new_both, ((importance_char[id_in_char] + importance_pseu[id_in_pseu]) / 2.0).reshape(1, 146, 48) ))
    elif (mni in successful_mnis_char_list) and (mni not in successful_mnis_pseu_list):
        only_in_char = np.vstack((only_in_char, mni))
        id_in_char = np.where(successful_mnis_char == mni)[0][0]
        importance_new_char = np.vstack((importance_new_char, importance_char[id_in_char].reshape(1, 146, 48)))
    elif (mni not in successful_mnis_char_list) and (mni in successful_mnis_pseu_list):
        only_in_pseu = np.vstack((only_in_pseu, mni))
        id_in_pseu = np.where(successful_mnis_pseu == mni)[0][0]
        importance_new_pseu = np.vstack((importance_new_pseu, importance_pseu[id_in_pseu].reshape(1, 146, 48)))
    else:
        print "Error: this can't be"

# important regions
mean_importance_both = np.mean(importance_new_both, axis=0)
mean_importance_char = np.mean(importance_new_char, axis=0)
mean_importance_pseu = np.mean(importance_new_pseu, axis=0)

mean_importance_both /= np.max(mean_importance_both)
mean_importance_char /= np.max(mean_importance_char)
mean_importance_pseu /= np.max(mean_importance_pseu)

significant_both = mean_importance_both > np.mean(mean_importance_both) + np.std(mean_importance_both) * 1.0
significant_char = mean_importance_char > np.mean(mean_importance_char) + np.std(mean_importance_char) * 1.0
significant_pseu = mean_importance_pseu > np.mean(mean_importance_pseu) + np.std(mean_importance_pseu) * 1.0

colormap = np.zeros((146, 48))
colormap[np.where(significant_char > 0)] =   2
colormap[np.where(significant_pseu > 0)] =  -1
colormap[np.where(significant_both > 0)] =   1
all_mnis = np.vstack((in_both, only_in_char, only_in_pseu))
triptych_fitfdiff(colormap, all_mnis, ['black'] * len(in_both) + ['red'] * len(only_in_char) + ['blue'] * len(only_in_pseu),
                  {'black': 0.5, 'red': 0.5, 'blue': 0.5}, ["%s/%d_%s-vs-%d_%s.png" % (OUTDIR, 6, categories[6], 5, categories[5])],
                  title="importance unique to %s (red), %s (blue)\nor common to both (gray)" % (categories[6], categories[5]), lines=True)
