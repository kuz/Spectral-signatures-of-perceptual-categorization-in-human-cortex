import numpy as np

# parameters
FREQSET = 'FT'
INDIR = '../../Outcome/Single Probe Classification/%s/Importances' % FREQSET

# known quantities
n_all_probes = 11321
n_responsive_probes = 11095

# predictive (successful) probes per category
all_predictive_mnis = np.empty((0, 3))
categories = ['house', 'visage', 'animal', 'scene', 'tool', 'pseudoword', 'characters', 'scrambled']
for cid, category in enumerate(categories):
    successful_mnis = np.load('%s/%s' % (INDIR, 'FT_successful_mnis_ctg%d.npy' % cid))
    drop_idx = np.unique(np.where(successful_mnis == [0, 0, 0])[0])
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)
    all_predictive_mnis = np.vstack((all_predictive_mnis, successful_mnis))
n_predictive_probes = np.unique(all_predictive_mnis, axis=0).shape[0]

print '\nStatistical summary\n-------------------'
print 'Total probes:', n_all_probes
print 'Responsive probes:', n_responsive_probes
print 'Predictive probes:', n_predictive_probes
print