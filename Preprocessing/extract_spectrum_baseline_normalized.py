import os
import numpy as np
import scipy.io as sio

# parameters
DATADIR = '../../Data/Intracranial/Processed'
featureset = 'ft_4hz150_LFP_8c_artif_bipolar_BA_responsive'

# lists
probelist = sorted(os.listdir('%s/%s' % (DATADIR, featureset)))
for i, filename in enumerate(probelist):
    if i % 100 == 0:
        print i
    sname = filename.replace('.mat', '')
    (sname, pid) = sname.split('-')
    s = sio.loadmat('%s/%s/%s' % (DATADIR, featureset, filename))
    normalizer = np.mean(s['ft'][:, :, 0:14], 2)
    normalizer = np.repeat(normalizer[:, :, np.newaxis], 48, axis=2)
    log_baseline_normalized = np.log(s['ft'] / normalizer)
    log_baseline_normalized[np.isnan(log_baseline_normalized)] = 0.0
    np.save('%s/normalized_%s/%s-%d.npy' % (DATADIR, featureset, sname, int(pid)), log_baseline_normalized)

 