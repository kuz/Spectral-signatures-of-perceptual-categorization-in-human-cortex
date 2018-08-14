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
subjlist = sorted(os.listdir('../../Outcome/Single Probe Classification/%s/Predictions' % FREQSET))
n_subjects = len(subjlist)
scores_spc = np.load('%s/../scores_sid_pid_cat.npy' % INDIR).item()

# global mono- and polypredictive probes
all_importance_mono = None
all_importance_poly = None
all_mnis_mono = np.empty((0, 3))
all_mnis_poly = np.empty((0, 3))

# competing categories
cid_a = 5
cid_b = 6

# load data
importance_a = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid_a))
successful_probes_a = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid_a))
successful_mnis_a = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid_a))
drop_idx_a = np.unique(np.where(successful_mnis_a == [0, 0, 0])[0])
importance_a = np.delete(importance_a, drop_idx_a, 0)
successful_probes_a = np.delete(successful_probes_a, drop_idx_a, 0)
successful_mnis_a = np.delete(successful_mnis_a, drop_idx_a, 0)

importance_b = np.load('%s/%s' % (INDIR, 'FT_feature_importances_ctg%d.npy' % cid_b))
successful_probes_b = np.load('%s/%s' % (INDIR, 'FT_successful_probes_ctg%d.npy' % cid_b))
successful_mnis_b = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid_b))
drop_idx_b = np.unique(np.where(successful_mnis_b == [0, 0, 0])[0])
importance_b = np.delete(importance_b, drop_idx_b, 0)
successful_probes_b = np.delete(successful_probes_b, drop_idx_b, 0)
successful_mnis_b = np.delete(successful_mnis_b, drop_idx_b, 0)

# not sigma-different importance
importance_threshold_a = np.mean(importance_a) + np.std(importance_a)
importance_threshold_b = np.mean(importance_b) + np.std(importance_b)
most_importantce_a = np.array(np.mean(importance_a, axis = 0) > importance_threshold_a, dtype=np.int8)
most_importantce_b = np.array(np.mean(importance_b, axis = 0) > importance_threshold_b, dtype=np.int8)
most_importantce = np.array(most_importantce_a + most_importantce_b >= 1, dtype=np.int8)

# sigma difference
sigma = 4.0
mean_importance_a = np.mean(importance_a, axis=0)
mean_importance_b = np.mean(importance_b, axis=0)
mean_importance_a /= np.max(mean_importance_a)
mean_importance_b /= np.max(mean_importance_b)
diff = mean_importance_a - mean_importance_b
absdiff = np.abs(diff)
significant_diff = absdiff > np.std(absdiff) * sigma

diffmap = most_importantce
diffmap[np.where(diff * significant_diff > 0)] =  2
diffmap[np.where(diff * significant_diff < 0)] = -1
both_mnis = np.vstack((successful_mnis_a, successful_mnis_b))
triptych_fitfdiff(diffmap, both_mnis, ['red'] * len(successful_mnis_a) + ['blue'] * len(successful_mnis_b),
                  {'red': 0.5, 'blue': 0.5}, ["%s/%d_%s-vs-%d_%s.png" % (OUTDIR, cid_a, categories[cid_a], cid_b, categories[cid_b])],
                  title="%d sigma difference between %s and %s" % (sigma, categories[cid_a], categories[cid_b]), lines=True)
