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
    successful_areas = np.load('%s/%s' % (INDIR, 'FT_successful_areas_ctg%d.npy' % cid))
    drop_idx = np.unique(np.where(successful_mnis == [0, 0, 0])[0])
    successful_mnis = np.delete(successful_mnis, drop_idx, 0)
    successful_areas = np.delete(successful_areas, drop_idx, 0)
    all_predictive_mnis = np.vstack((all_predictive_mnis, successful_mnis))

    # Within the primary visual cortex, BA 17 and 18, the scrambled was the stimulus that
    # elicited most predictive probes amongst all stimulus categories
    print category[:4], 'in 17, 18 --', np.sum(successful_areas == 17) + np.sum(successful_areas == 18)

    # Probes predictive of \texttt{faces} were mostly concentrated in BA19, BA37 and BA20
    if category == 'visage':
        print category[:4], 'in 19, 37, 10 --', np.sum(successful_areas == 19) + np.sum(successful_areas == 37) + np.sum(successful_areas == 20), 'out of', successful_areas.shape[0]


n_predictive_probes = np.unique(all_predictive_mnis, axis=0).shape[0]

print '\nStatistical summary\n-------------------'
print 'Total probes:', n_all_probes
print 'Responsive probes:', n_responsive_probes
print 'Predictive probes:', n_predictive_probes
print